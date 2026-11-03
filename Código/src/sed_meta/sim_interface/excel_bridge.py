"""
Fase 2 — interface com o simulador, via cockpit Excel.

A troca de dados com o ProModel é feita por uma planilha .xlsx
("cockpit") no formato:

    uma linha = uma execução completa (um ponto do DoE x uma replicação)

    [ ---- ENTRADA (esquerda) ---- ] [ ---- SAÍDA (direita) ---- ]
    point_id | replication_id | unique_id | <params...> | <KPIs...>
                                                     (vazios)

O ProModel é configurado POR FORA deste código (configuração manual,
fora do escopo do Python) para varrer a planilha sequencialmente: pega a
próxima linha ainda não executada, lê os parâmetros de entrada daquela
linha, roda a simulação e escreve os KPIs na MESMA linha antes de
avançar. Por isso as colunas de saída nascem vazias (NaN -> célula em
branco): célula em branco é o marcador de "ainda não executado".

Posição no pipeline:

    parameter_space.generate_design()      -> N pontos
    replication.build_execution_plan()     -> N x R linhas (uma por execução)
    excel_bridge.build_cockpit_table()     -> + colunas de KPI vazias
    excel_bridge.write_cockpit_excel()     -> .xlsx entregue ao ProModel
    ................ ProModel roda e preenche a planilha ................
    excel_bridge.read_cockpit_excel()      -> .xlsx de volta em DataFrame
    excel_bridge.extract_results()         -> `raw`, TODAS as linhas + status
    excel_bridge.select_complete_runs()    -> recorte explícito das completas
    replication.aggregate_replications() / compute_noise_ceiling()

Duas decisões de formato que valem registro, porque afetam quem lê o
arquivo do outro lado (ProModel) e a fidelidade dos tipos:

1. A aba de dados tem UMA ÚNICA linha de cabeçalho (linha 1), sem faixa
   mesclada "Entrada"/"Saída" acima dela. Uma segunda linha de cabeçalho
   deslocaria os dados e tornaria ambígua a leitura pelo ProModel. A
   distinção visual entrada/saída é feita só por preenchimento de cor e
   negrito NAS CÉLULAS do cabeçalho — efeito visual sem deslocar nada.

2. O arquivo carrega uma aba oculta de esquema (`_sed_meta_schema`) com o
   dtype original de cada coluna. Sem isso o round-trip pelo Excel perde
   tipo (int64 vira float64, category vira texto solto, e a ordem de uma
   category ordenada — ex.: prioridade baixa < media < alta — some).
   Como read_cockpit_excel() recebe apenas o `path`, o esquema precisa
   viajar dentro do próprio arquivo. Se a aba de esquema não existir
   (planilha montada à mão, ou salva por outra ferramenta que a perdeu),
   a leitura cai num modo best-effort documentado em _restore_sem_esquema().

REGRA DO PROJETO — NADA É DELETADO SILENCIOSAMENTE: nenhuma linha do
cockpit desaparece na leitura. extract_results() devolve TODAS as
execuções e MARCA cada uma com um status (completa / parcial / pendente
/ invalida) na coluna `status_execucao`; quem consome decide o que
fazer. Selecionar só as completas é um passo explícito e separado —
select_complete_runs() —, que loga o que está deixando de fora. Toda
situação anômala (KPI não numérico, linha parcial, dtype que não pôde
ser restaurado, coluna sumida da planilha) é sinalizada por logging em
nível WARNING, além de aparecer no status.

LIMITE CONHECIDO (float): o Excel guarda no máximo 15 dígitos
significativos, então um parâmetro float não volta bit a bit idêntico ao
que o DoE gerou — ex.: 11.108535497068328 volta 11.10853549706833, erro
relativo da ordem de 1e-16 (1 ULP). Isso é limitação do formato, não do
código, e é inofensivo aqui por dois motivos: (a) o agrupamento das
replicações é por `point_id`, não por valor de parâmetro; (b) o
arredondamento é determinístico, então todas as replicações de um mesmo
ponto continuam com exatamente o mesmo valor. Só não use `.equals()` /
`==` exato para comparar o plano original com o lido de volta — use
np.isclose(). Se algum dia for preciso reproduzir bit a bit, o caminho é
guardar o plano original em Parquet/CSV ao lado do .xlsx.
"""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Nomes de aba: a de dados vem primeiro no arquivo (é a que o ProModel lê);
# a de esquema é oculta e serve só ao round-trip de dtypes deste módulo.
DATA_SHEET = "cockpit"
SCHEMA_SHEET = "_sed_meta_schema"

# Coluna de status acrescentada por extract_results(): é o que permite
# manter TODAS as linhas no resultado sem confundir executado com pendente.
STATUS_COL = "status_execucao"
STATUS_COMPLETA = "completa"    # todos os KPIs preenchidos e numéricos
STATUS_PARCIAL = "parcial"      # alguns KPIs preenchidos, outros em branco
STATUS_PENDENTE = "pendente"    # nenhum KPI preenchido (ainda não rodou)
STATUS_INVALIDA = "invalida"    # KPI preenchido com algo que não é número

# Colunas de bookkeeping criadas por replication.build_execution_plan().
# Identificam a execução no cockpit, mas NÃO são
# parâmetros do modelo — ver nota em extract_results().
ID_COLS = ("point_id", "replication_id", "unique_id")

# Tons claros padrão do Excel; só para o operador bater o olho e saber
# onde termina o que ele fornece e começa o que o ProModel preenche.
_FILL_ENTRADA = "DDEBF7"  # azul claro
_FILL_SAIDA = "FCE4D6"    # laranja claro


# ---------------------------------------------------------------------
# 1. Montagem da tabela do cockpit (pura, sem I/O)
# ---------------------------------------------------------------------

def build_cockpit_table(plan: pd.DataFrame, kpi_names: list[str]) -> pd.DataFrame:
    """Monta a tabela plana do cockpit a partir do plano de execução.

    Parâmetros
    ----------
    plan : saída de replication.build_execution_plan() — N x R linhas,
           colunas point_id, replication_id, unique_id + colunas de parâmetro.
           A granularidade "uma linha por execução" dessa função é
           exatamente a granularidade de uma linha do cockpit.
    kpi_names : nomes das colunas de saída que o ProModel vai preencher.

    Retorna
    -------
    pd.DataFrame com as colunas de `plan` NA MESMA ORDEM em que vieram
    (bloco de entrada), seguidas de uma coluna por KPI, todas NaN
    (bloco de saída). NaN aqui é semântico: "ainda não executado".
    """
    if not kpi_names:
        raise ValueError("kpi_names não pode ser vazio: o cockpit precisa de colunas de saída.")

    duplicados = [k for k in set(kpi_names) if kpi_names.count(k) > 1]
    if duplicados:
        raise ValueError(f"kpi_names tem nomes repetidos: {sorted(duplicados)}")

    colisoes = [k for k in kpi_names if k in plan.columns]
    if colisoes:
        raise ValueError(
            f"KPIs colidem com colunas de entrada do plano: {colisoes}. "
            "Renomeie o KPI ou o parâmetro — o cockpit não pode ter coluna ambígua."
        )

    cockpit = plan.copy()
    for kpi in kpi_names:
        # float64 explícito: coluna vazia que vai receber número do ProModel
        cockpit[kpi] = np.full(len(cockpit), np.nan, dtype="float64")
    return cockpit


# ---------------------------------------------------------------------
# 2. Escrita do .xlsx (entrega ao ProModel)
# ---------------------------------------------------------------------

def write_cockpit_excel(
    cockpit_df: pd.DataFrame,
    path: str,
    entrada_cols: list[str],
    saida_cols: list[str],
) -> None:
    """Escreve o cockpit em disco como .xlsx.

    A ordem das colunas no arquivo é exatamente `entrada_cols + saida_cols`
    (entrada à esquerda, saída à direita) — é esse contrato posicional que
    a configuração manual do ProModel enxerga.

    Junto vai uma aba oculta de esquema, para que read_cockpit_excel()
    consiga devolver os dtypes originais (ver docstring do módulo).
    """
    faltando = [c for c in (*entrada_cols, *saida_cols) if c not in cockpit_df.columns]
    if faltando:
        raise ValueError(f"Colunas pedidas que não existem no DataFrame: {faltando}")

    sobrando = [c for c in cockpit_df.columns if c not in (*entrada_cols, *saida_cols)]
    if sobrando:
        raise ValueError(
            f"Colunas do DataFrame não classificadas como entrada nem saída: {sobrando}. "
            "Toda coluna do cockpit precisa estar de um dos dois lados."
        )

    ordenado = cockpit_df[[*entrada_cols, *saida_cols]]
    schema = _build_schema_frame(ordenado, entrada_cols, saida_cols)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        ordenado.to_excel(writer, sheet_name=DATA_SHEET, index=False)
        schema.to_excel(writer, sheet_name=SCHEMA_SHEET, index=False)
        _format_header(writer, ordenado, entrada_cols, saida_cols)
        writer.book[SCHEMA_SHEET].sheet_state = "hidden"


def _build_schema_frame(
    df: pd.DataFrame, entrada_cols: list[str], saida_cols: list[str]
) -> pd.DataFrame:
    """Serializa o dtype de cada coluna numa tabela gravável em planilha.

    Para colunas category guarda também os níveis e se são ordenados —
    é o que permite reconstruir uma variável ordinal (ex.: prioridade
    baixa < media < alta) do jeito que parameter_space._cast_dtypes()
    a criou.
    """
    linhas = []
    for col in df.columns:
        serie = df[col]
        categorias, ordenada = None, None
        if isinstance(serie.dtype, pd.CategoricalDtype):
            # .tolist() converte np.int64 -> int nativo, senão json.dumps quebra
            categorias = json.dumps(serie.cat.categories.tolist())
            ordenada = bool(serie.cat.ordered)
        linhas.append({
            "coluna": col,
            "papel": "entrada" if col in entrada_cols else "saida",
            "dtype": str(serie.dtype),
            "categorias_json": categorias,
            "ordenada": ordenada,
        })
    return pd.DataFrame(linhas)


def _format_header(writer, df: pd.DataFrame, entrada_cols: list[str], saida_cols: list[str]) -> None:
    """Formatação leve do cabeçalho: cor por bloco, negrito, largura e
    painel congelado. Só mexe na linha 1 — nenhuma linha é inserida, para
    não deslocar os dados que o ProModel lê."""
    from openpyxl.styles import Alignment, Font, PatternFill

    sheet = writer.book[DATA_SHEET]
    fill_entrada = PatternFill("solid", fgColor=_FILL_ENTRADA)
    fill_saida = PatternFill("solid", fgColor=_FILL_SAIDA)

    for idx, col in enumerate(df.columns, start=1):
        cell = sheet.cell(row=1, column=idx)
        cell.fill = fill_entrada if col in entrada_cols else fill_saida
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        sheet.column_dimensions[cell.column_letter].width = max(len(str(col)) + 2, 12)

    sheet.freeze_panes = "A2"


# ---------------------------------------------------------------------
# 3. Leitura do .xlsx (de volta do ProModel)
# ---------------------------------------------------------------------

def read_cockpit_excel(path: str) -> pd.DataFrame:
    """Lê o cockpit de volta, restaurando os dtypes originais.

    O Excel não guarda dtype de pandas: tudo volta como float64 ou texto.
    Quando a aba de esquema escrita por write_cockpit_excel() está
    presente, os dtypes são restaurados exatamente (inclusive category
    ordenada). Quando não está, cai no best-effort de
    _restore_sem_esquema().

    Atenção: dtype e valor voltam iguais, mas float volta arredondado a 15
    dígitos significativos (limite do Excel) — ver "LIMITE CONHECIDO" no
    docstring do módulo. Compare com np.isclose(), não com `==`.
    """
    with pd.ExcelFile(path, engine="openpyxl") as book:
        nome_dados = DATA_SHEET if DATA_SHEET in book.sheet_names else next(
            (s for s in book.sheet_names if s != SCHEMA_SHEET), book.sheet_names[0]
        )
        df = pd.read_excel(book, sheet_name=nome_dados)
        schema = (
            pd.read_excel(book, sheet_name=SCHEMA_SHEET)
            if SCHEMA_SHEET in book.sheet_names
            else None
        )

    if schema is None:
        logger.warning(
            "Planilha %s sem a aba '%s': os dtypes serão inferidos em modo "
            "best-effort (category/ordinal não são recuperáveis assim). "
            "Ver _restore_sem_esquema().", path, SCHEMA_SHEET,
        )
        return _restore_sem_esquema(df)
    return _restore_com_esquema(df, schema)


def _restore_com_esquema(df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    sumidas = [c for c in schema["coluna"] if c not in df.columns]
    if sumidas:
        # não é papel daqui decidir o que fazer com isso, mas passar
        # batido seria exatamente o silêncio proibido
        logger.warning(
            "Colunas que existiam quando o cockpit foi gerado e não estão "
            "mais na planilha: %s. Foram apagadas ou renomeadas no Excel?",
            sumidas,
        )
    extras = [c for c in df.columns if c not in set(schema["coluna"])]
    if extras:
        logger.warning(
            "Colunas presentes na planilha que não estavam no cockpit gerado: "
            "%s. Serão lidas com o dtype que o pandas inferir.", extras,
        )

    for _, linha in schema.iterrows():
        col = linha["coluna"]
        if col not in df.columns:
            continue

        if isinstance(linha["categorias_json"], str):
            categorias = json.loads(linha["categorias_json"])
            alvo = pd.CategoricalDtype(categories=categorias, ordered=bool(linha["ordenada"]))
        else:
            alvo = linha["dtype"]

        df[col] = _cast_seguro(df[col], alvo, col)
    return df


def _cast_seguro(serie: pd.Series, alvo, nome_col: str) -> pd.Series:
    """Converte para `alvo`, mas desiste se a conversão destruir dado.

    Dois casos reais: (a) o ProModel/operador escreveu algo fora dos
    níveis de uma category — o astype viraria NaN silenciosamente;
    (b) a coluna de KPI ainda está vazia e o dtype alvo é int64 — o
    astype levanta. Nos dois casos devolvemos o dado como veio, para não
    destruir nada, e avisamos: dtype inesperado é anomalia e não pode
    passar em silêncio.
    """
    try:
        convertida = serie.astype(alvo)
    except (ValueError, TypeError) as erro:
        logger.warning(
            "Coluna '%s': não foi possível restaurar o dtype %r (%s). "
            "Mantida como %s, sem perda de valor.",
            nome_col, alvo, erro, serie.dtype,
        )
        return serie

    perdidos = int(convertida.isna().sum() - serie.isna().sum())
    if perdidos > 0:
        logger.warning(
            "Coluna '%s': restaurar o dtype %r transformaria %d valor(es) em "
            "NaN (valor fora do esperado na planilha). Conversão abortada, "
            "coluna mantida como %s.",
            nome_col, alvo, perdidos, serie.dtype,
        )
        return serie
    return convertida


def _restore_sem_esquema(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort para planilha sem a aba de esquema.

    Regra única e conservadora: coluna float sem nenhum vazio e com todos
    os valores inteiros volta a int64 (é o caso de point_id, replication_id,
    unique_id e parâmetros inteiros, que o Excel devolve como 0.0, 1.0, 2.0).
    Não há como distinguir aqui um float genuíno que por acaso caiu em
    valores inteiros — por isso o caminho com esquema é o preferido.
    """
    for col in df.columns:
        serie = df[col]
        if serie.dtype.kind != "f" or serie.isna().any() or serie.empty:
            continue
        if np.array_equal(serie.to_numpy(), np.round(serie.to_numpy())):
            df[col] = serie.astype("int64")
    return df


# ---------------------------------------------------------------------
# 4. Ponte de volta para o código já validado de replication.py
# ---------------------------------------------------------------------

def extract_results(cockpit_df: pd.DataFrame, kpi_names: list[str]) -> pd.DataFrame:
    """Converte o cockpit lido no formato `raw`, MARCANDO cada execução.

    Não filtra nada. Toda linha do cockpit sai no resultado, na mesma
    ordem, com a coluna `status_execucao` dizendo em que estado ela está:

      completa  todos os KPIs preenchidos com número -> pronta para agregar
      parcial   alguns KPIs preenchidos, outros em branco -> execução
                interrompida no meio, ou KPI que o ProModel não escreveu
      pendente  nenhum KPI preenchido -> ainda não rodou (estado normal
                de um cockpit em andamento)
      invalida  algum KPI preenchido com algo que não é número (texto,
                "#N/D", célula suja) -> precisa de inspeção humana

    Os três estados que não são "completa" também vão para o log em
    WARNING (pendente vai em INFO: é progresso normal, não anomalia),
    com point_id/replication_id e a linha correspondente na planilha,
    para dar para achar a célula no .xlsx.

    Formato de saída (o `raw` que replication.py consome): point_id,
    replication_id, unique_id, colunas de parâmetro, colunas de KPI e
    status_execucao. As colunas de KPI vêm convertidas para numérico —
    valor não numérico vira NaN na coluna, mas o valor original continua
    no .xlsx e é reportado no log, e a linha fica marcada `invalida`.

    Cuidado ao derivar `param_cols` para aggregate_replications(): use as
    colunas do design, ou exclua explicitamente ID_COLS + kpi_names +
    STATUS_COL. point_id/replication_id/unique_id são bookkeeping, não fatores
    do modelo — foram mantidos aqui porque sem eles não há como
    rastrear, no .xlsx, qual execução ficou parcial ou inválida.

    IMPORTANTE: para agregar, passe antes por select_complete_runs(). A
    seleção é explícita de propósito — este módulo não decide sozinho
    jogar linha fora (ver "REGRA DO PROJETO" no docstring do módulo).
    """
    faltando = [k for k in kpi_names if k not in cockpit_df.columns]
    if faltando:
        raise ValueError(f"Colunas de KPI ausentes na planilha: {faltando}")
    if "point_id" not in cockpit_df.columns:
        raise ValueError("Planilha sem coluna 'point_id' — não dá para agrupar replicações.")

    # índice normalizado logo de saída: o log identifica cada execução pela
    # linha correspondente da planilha (idx + 2), e isso só fecha se o
    # índice for 0..n-1 na mesma ordem em que as linhas estão no .xlsx
    original = cockpit_df.reset_index(drop=True)
    df = original.copy()

    vazio = pd.DataFrame(
        {kpi: original[kpi].isna() for kpi in kpi_names}, index=df.index
    )
    for kpi in kpi_names:
        # Excel devolve número digitado como texto ("12.5") em célula
        # formatada como texto; to_numeric normaliza isso. O que não é
        # número vira NaN — mas, diferente de antes, isso não some: a
        # linha é marcada `invalida` e o valor original vai para o log.
        df[kpi] = pd.to_numeric(df[kpi], errors="coerce")

    numerico = df[list(kpi_names)].notna()
    # preenchido mas não numérico = célula suja
    invalido = (~vazio) & (~numerico)

    status = pd.Series(STATUS_PARCIAL, index=df.index, dtype="object")
    status[numerico.all(axis=1)] = STATUS_COMPLETA
    status[vazio.all(axis=1)] = STATUS_PENDENTE
    status[invalido.any(axis=1)] = STATUS_INVALIDA  # tem precedência
    df[STATUS_COL] = status

    param_cols = [
        c for c in original.columns
        if c not in ID_COLS and c not in kpi_names and c != STATUS_COL
    ]
    raw = df[[*_id_cols_presentes(df), *param_cols, *kpi_names, STATUS_COL]]

    _logar_status(raw, kpi_names, original, invalido)
    return raw


def _id_cols_presentes(df: pd.DataFrame) -> list[str]:
    """Colunas de ID_COLS que existem no DataFrame, na ordem canônica.

    point_id é obrigatório (validado antes); replication_id e unique_id podem
    faltar se o `raw` veio de um DataFrame sintético de teste em vez do
    cockpit completo.
    """
    return [c for c in ID_COLS if c in df.columns]


def _logar_status(
    raw: pd.DataFrame,
    kpi_names: list[str],
    original: pd.DataFrame,
    invalido: pd.DataFrame,
) -> None:
    """Faz barulho proporcional à anomalia. Nada aqui altera dados."""
    contagem = raw[STATUS_COL].value_counts().to_dict()
    logger.info(
        "Cockpit lido: %d execuções — %s",
        len(raw),
        ", ".join(f"{k}={v}" for k, v in sorted(contagem.items())) or "vazio",
    )

    for status, nivel in ((STATUS_PARCIAL, logging.WARNING),
                          (STATUS_INVALIDA, logging.WARNING)):
        afetadas = raw.index[raw[STATUS_COL] == status]
        if len(afetadas) == 0:
            continue
        logger.log(
            nivel,
            "%d execução(ões) com status '%s' — mantidas no resultado e NÃO "
            "agregáveis até serem resolvidas: %s",
            len(afetadas), status,
            "; ".join(_descrever_linha(raw, i, kpi_names) for i in afetadas[:10])
            + (" ..." if len(afetadas) > 10 else ""),
        )

    # o valor original que não virou número é a informação mais útil para
    # achar a célula suja no .xlsx — por isso vai explícito no log
    for col in invalido.columns:
        for idx in invalido.index[invalido[col]]:
            logger.warning(
                "Valor não numérico em '%s', %s: %r (virou NaN na coluna, "
                "valor original preservado na planilha).",
                col, _descrever_linha(raw, idx, kpi_names), original.at[idx, col],
            )

    pendentes = int((raw[STATUS_COL] == STATUS_PENDENTE).sum())
    if pendentes:
        logger.info(
            "%d execução(ões) ainda não rodadas (status '%s') — mantidas no "
            "resultado, aguardando o ProModel.", pendentes, STATUS_PENDENTE,
        )


def _descrever_linha(raw: pd.DataFrame, idx, kpi_names: list[str]) -> str:
    """Identifica a execução de um jeito que dê para achar no .xlsx: ids
    de execução e a linha da planilha (o +2 é cabeçalho + base-1 do
    Excel). O mapeamento supõe o índice normalizado por extract_results()
    e a ordem original das linhas — reordenar o DataFrame antes invalida
    o número da linha, não os ids."""
    partes = [f"linha_xlsx={idx + 2}"]
    for col in _id_cols_presentes(raw):
        partes.append(f"{col}={raw.at[idx, col]}")
    return "(" + ", ".join(partes) + ")"


def select_complete_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Seleciona SÓ as execuções completas, avisando o que ficou de fora.

    Este é o ponto único e explícito onde linhas deixam de seguir adiante
    no pipeline — por isso é uma função separada, com nome próprio e log
    próprio, em vez de um filtro escondido dentro de extract_results().
    Chame-a imediatamente antes de replication.aggregate_replications() /
    compute_noise_ceiling(), que assumem execuções completas.

    O que sai daqui é um recorte: o DataFrame original com TODAS as
    linhas continua nas mãos de quem chamou (e a planilha, que é o
    artefato de registro, nunca é tocada).
    """
    if STATUS_COL not in raw.columns:
        raise ValueError(
            f"DataFrame sem a coluna '{STATUS_COL}' — passe a saída de "
            "extract_results(), que é quem marca o status de cada execução."
        )

    completas = raw[raw[STATUS_COL] == STATUS_COMPLETA]
    descartadas = len(raw) - len(completas)
    if descartadas:
        fora = raw.loc[raw[STATUS_COL] != STATUS_COMPLETA, STATUS_COL].value_counts()
        logger.warning(
            "select_complete_runs: seguindo com %d de %d execuções; %d fora "
            "deste recorte (%s). Nada foi apagado — as linhas continuam no "
            "DataFrame de origem e na planilha.",
            len(completas), len(raw), descartadas,
            ", ".join(f"{k}={v}" for k, v in sorted(fora.items())),
        )
    return completas.reset_index(drop=True)




# ---------------------------------------------------------------------
# 5. Cockpit no layout do ProModel (o que vai de fato para a interface)
# ---------------------------------------------------------------------
#
# POR QUE UM SEGUNDO FORMATO
#
# write_cockpit_excel()/read_cockpit_excel() acima usam um cabeçalho único
# na linha 1. Os cockpits que a Macro do ProModel lê têm outro layout, e é
# ele que manda na interface — substitui, para os arquivos entregues ao
# ProModel, a regra do cabeçalho único do item 1 do docstring:
#
#     linha 1   vazia
#     linha 2   faixas     [---- Entrada ----] [---- Saída ----] [Identificação]
#     linha 3   A vazia | Item | <entradas> | <saídas> | point_id | replication_id | unique_id
#     linha 4+          | Teste n | valores | (vazias) | ids
#
# O padrão visual (cores, bordas, faixas) está nas constantes _PM_* abaixo:
# nenhum arquivo-modelo é lido, e todo cenário usa o mesmo padrão.
# `Item` = "Teste n", n = 1..N = posição da execução: é o ponteiro da Macro.
# As colunas de identificação ficam À DIREITA das saídas, com cor própria,
# para não deslocar o que a Macro lê (Item na coluna B, entradas a partir
# da C, saídas logo depois).
#
# As funções antigas continuam intactas: os testes e o notebook do pipeline
# dependem delas.

PROMODEL_DATA_FIRST_ROW = 4     # linha 1 vazia, 2 faixas, 3 cabeçalho
_PM_FILL_ITEM = "FF002060"      # azul-marinho do Item/"Teste n"
_PM_FILL_ENTRADA = "FF00B050"   # faixa verde
_PM_FILL_SAIDA = "FFFF0000"     # faixa vermelha
_PM_FILL_VALOR = "FFFFFF00"     # células de entrada/saída, amarelas
_PM_FILL_ID = "FFBFBFBF"        # faixa e cabeçalho de identificação, cinza
_PM_FILL_ID_VALOR = "FFD9D9D9"  # células de identificação, cinza claro
_PM_FAIXA_ID = "Identificação"


def write_promodel_cockpit(
    cockpit_df: pd.DataFrame,
    path: str,
    input_cols: list[str],
    output_cols: list[str],
    *,
    sheet_name: str = DATA_SHEET,
    item_column: str = "Item",
    item_prefix: str = "Teste ",
) -> None:
    """Escreve o cockpit no layout do ProModel.

    Parâmetros
    ----------
    cockpit_df : saída de build_cockpit_table() — point_id, replication_id,
                 unique_id, parâmetros, KPIs vazios; uma linha por execução,
                 NA ORDEM em que o ProModel deve rodar.
    input_cols, output_cols : ordem das colunas de entrada e de saída no
                 arquivo (a ordem que a Macro lê).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    faltando = [c for c in (*ID_COLS, *input_cols, *output_cols) if c not in cockpit_df.columns]
    if faltando:
        raise ValueError(f"Colunas pedidas que não existem no DataFrame: {faltando}")
    sobrando = [c for c in cockpit_df.columns if c not in (*ID_COLS, *input_cols, *output_cols)]
    if sobrando:
        raise ValueError(
            f"Colunas do DataFrame não classificadas como entrada nem saída: {sobrando}."
        )
    reservados = {item_column, *ID_COLS}.intersection((*input_cols, *output_cols))
    if reservados:
        raise ValueError(f"Entrada/saída com nome reservado do layout: {sorted(reservados)}")

    df = cockpit_df.reset_index(drop=True)
    itens = [f"{item_prefix}{i}" for i in range(1, len(df) + 1)]

    # esquema: nome no arquivo -> nome interno, papel e dtype
    linhas_schema = [{"coluna": item_column, "interno": None, "papel": "item",
                      "dtype": "object", "categorias_json": None, "ordenada": None}]
    for papel, cols in (("entrada", input_cols), ("saida", output_cols), ("id", ID_COLS)):
        for col in cols:
            serie = df[col]
            categorias, ordenada = None, None
            if isinstance(serie.dtype, pd.CategoricalDtype):
                categorias = json.dumps(serie.cat.categories.tolist())
                ordenada = bool(serie.cat.ordered)
            linhas_schema.append({"coluna": col, "interno": col, "papel": papel,
                                  "dtype": str(serie.dtype), "categorias_json": categorias,
                                  "ordenada": ordenada})
    schema = pd.DataFrame(linhas_schema)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    fina = Side(style="thin")
    borda = Border(left=fina, right=fina, top=fina, bottom=fina)
    centro = Alignment(horizontal="center", vertical="center")
    fill = {k: PatternFill("solid", fgColor=v) for k, v in {
        "item": _PM_FILL_ITEM, "entrada": _PM_FILL_ENTRADA, "saida": _PM_FILL_SAIDA,
        "valor": _PM_FILL_VALOR, "id": _PM_FILL_ID, "id_valor": _PM_FILL_ID_VALOR}.items()}
    branco_negrito = Font(bold=True, color="FFFFFFFF")

    # colunas: A vazia, B = Item, C.. = entradas, depois saídas, depois ids
    col_entrada_fim = 2 + len(input_cols)       # a faixa Entrada começa no Item (B)
    col_saida_ini, col_saida_fim = col_entrada_fim + 1, col_entrada_fim + len(output_cols)
    col_id_ini, col_id_fim = col_saida_fim + 1, col_saida_fim + len(ID_COLS)

    # linha 2 — faixas
    for rotulo, ini, fim, f in (("Entrada", 2, col_entrada_fim, "entrada"),
                                ("Saída", col_saida_ini, col_saida_fim, "saida"),
                                (_PM_FAIXA_ID, col_id_ini, col_id_fim, "id")):
        for j in range(ini, fim + 1):
            ws.cell(row=2, column=j).fill = fill[f]
        cel = ws.cell(row=2, column=ini, value=rotulo)
        cel.alignment = centro
        cel.font = Font(color="FF000000")
        if fim > ini:
            ws.merge_cells(start_row=2, start_column=ini, end_row=2, end_column=fim)

    # linha 3 — cabeçalho
    cabecalho = [item_column, *input_cols, *output_cols, *ID_COLS]
    for j, nome in enumerate(cabecalho, start=2):
        cel = ws.cell(row=3, column=j, value=nome)
        cel.border = borda
        if j == 2:
            cel.fill, cel.font, cel.alignment = fill["item"], branco_negrito, centro
        elif j >= col_id_ini:
            cel.fill, cel.font, cel.alignment = fill["id"], Font(bold=True), centro
        ws.column_dimensions[get_column_letter(j)].width = max(len(nome) + 2, 10)

    # linha 4+ — dados (célula vazia na saída = "ainda não executado")
    valores_entrada = [df[c].tolist() for c in input_cols]
    valores_id = [df[c].tolist() for c in ID_COLS]
    for i in range(len(df)):
        r = PROMODEL_DATA_FIRST_ROW + i
        cel = ws.cell(row=r, column=2, value=itens[i])
        cel.fill, cel.font, cel.alignment = fill["item"], branco_negrito, centro
        for k, coluna in enumerate(valores_entrada):
            cel = ws.cell(row=r, column=3 + k, value=_valor_excel(coluna[i]))
            cel.fill, cel.border = fill["valor"], borda
        for j in range(col_saida_ini, col_saida_fim + 1):
            cel = ws.cell(row=r, column=j)
            cel.fill, cel.border = fill["valor"], borda
        for k, coluna in enumerate(valores_id):
            cel = ws.cell(row=r, column=col_id_ini + k, value=_valor_excel(coluna[i]))
            cel.fill, cel.border, cel.alignment = fill["id_valor"], borda, centro

    ws.freeze_panes = ws.cell(row=PROMODEL_DATA_FIRST_ROW, column=3)

    aba = wb.create_sheet(SCHEMA_SHEET)
    aba.append(list(schema.columns))
    for linha in schema.itertuples(index=False):
        aba.append([_valor_excel(v) for v in linha])
    aba.sheet_state = "hidden"

    wb.save(path)


def _valor_excel(v):
    """numpy -> tipo nativo (openpyxl não aceita np.int64) e NaN -> célula vazia."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_promodel_cockpit(path: str, sheet_name: str = DATA_SHEET) -> pd.DataFrame:
    """Lê um cockpit no layout do ProModel e devolve o vocabulário interno.

    Saída: point_id, replication_id, unique_id, entradas, saídas — pronta
    para extract_results(df, output_cols) e select_complete_runs() sem
    nenhuma adaptação. Aceita .xlsx e .xlsm (o ProModel pode exigir .xlsm;
    openpyxl lê os dois — a macro em si é ignorada).

    Com a aba oculta _sed_meta_schema, os dtypes são restaurados exatamente.
    As colunas são achadas pelo NOME na linha 3 e os ids viajam na própria
    linha, então reordenar linhas no Excel não troca réplicas.

    Sem o esquema (cockpit montado à mão, ou salvo por ferramenta que perdeu
    a aba oculta), cai num best-effort documentado:
      - entradas entre Item e o início da faixa "Saída"; saídas daí até a
        faixa "Identificação" ou a primeira coluna sem cabeçalho (o que
        descarta lixo à direita, como as colunas BA:BC do exemplo nível 3);
      - ids pelas colunas point_id/replication_id/unique_id, se existirem;
        sem point_id, point_id = posição da linha; sem replication_id, ordem
        de aparição dentro do point_id;
      - dtypes pela regra de _restore_sem_esquema().

    Linhas sem Item são formatação vazia além do fim do plano e não entram;
    as demais entram TODAS (nada some em silêncio).
    """
    with pd.ExcelFile(path, engine="openpyxl") as book:
        if sheet_name not in book.sheet_names:
            raise ValueError(f"{path}: aba '{sheet_name}' não encontrada ({book.sheet_names}).")
        bruto = pd.read_excel(book, sheet_name=sheet_name, header=None)
        schema = (pd.read_excel(book, sheet_name=SCHEMA_SHEET)
                  if SCHEMA_SHEET in book.sheet_names else None)

    # header=None: coluna 0 = coluna A e linha 0 = linha 1, mesmo vazias
    cabecalho = bruto.iloc[2].tolist() if len(bruto) > 2 else []
    dados = bruto.iloc[PROMODEL_DATA_FIRST_ROW - 1:].reset_index(drop=True)

    if schema is not None and {"papel", "interno"}.issubset(schema.columns):
        return _ler_promodel_com_esquema(path, cabecalho, dados, schema)

    logger.warning(
        "Cockpit %s sem a aba '%s': leitura best-effort. Ver read_promodel_cockpit().",
        path, SCHEMA_SHEET,
    )
    return _ler_promodel_sem_esquema(path, bruto, cabecalho, dados)


def _ler_promodel_com_esquema(path, cabecalho, dados, schema) -> pd.DataFrame:
    posicao = {nome: j for j, nome in enumerate(cabecalho) if isinstance(nome, str)}
    sumidas = [c for c in schema["coluna"] if c not in posicao]
    if sumidas:
        raise ValueError(
            f"{path}: colunas do cockpit gerado que não estão mais na linha 3: {sumidas}. "
            "Foram apagadas ou renomeadas no Excel?"
        )

    item_col = schema.loc[schema["papel"] == "item", "coluna"].iloc[0]
    dados = dados[dados[posicao[item_col]].notna()].reset_index(drop=True)

    # ids primeiro, depois entradas e saídas — o vocabulário interno
    out = pd.DataFrame(index=range(len(dados)))
    for papel in ("id", "entrada", "saida"):
        for _, linha in schema[schema["papel"] == papel].iterrows():
            col = linha["interno"]
            if isinstance(linha["categorias_json"], str):
                alvo = pd.CategoricalDtype(json.loads(linha["categorias_json"]),
                                           ordered=bool(linha["ordenada"]))
            else:
                alvo = linha["dtype"]
            bruta = pd.Series(dados[posicao[linha["coluna"]]].to_numpy())
            out[col] = _cast_seguro(_numerico_se_possivel(bruta), alvo, col)
    return out


def _ler_promodel_sem_esquema(path, bruto, cabecalho, dados) -> pd.DataFrame:
    faixas = bruto.iloc[1].tolist() if len(bruto) > 1 else []
    try:
        j_item = next(j for j, v in enumerate(cabecalho) if v == "Item")
        j_saida = next(j for j, v in enumerate(faixas)
                       if isinstance(v, str) and v.strip() == "Saída")
    except StopIteration:
        raise ValueError(f"{path}: não achei 'Item' na linha 3 ou a faixa 'Saída' na linha 2.")

    # as saídas acabam na faixa Identificação, num id ou na 1ª coluna sem cabeçalho
    j_fim = j_saida
    while (j_fim < len(cabecalho) and isinstance(cabecalho[j_fim], str)
           and cabecalho[j_fim] not in ID_COLS
           and not (j_fim > j_saida and faixas[j_fim] == _PM_FAIXA_ID)):
        j_fim += 1

    dados = dados[dados[j_item].notna()].reset_index(drop=True)
    posicao = {nome: j for j, nome in enumerate(cabecalho) if isinstance(nome, str)}
    out = pd.DataFrame(index=range(len(dados)))
    for col in ID_COLS:
        if col in posicao:
            out[col] = _numerico_se_possivel(dados[posicao[col]]).to_numpy()
    if "point_id" not in out.columns:
        logger.warning("%s: sem coluna point_id — point_id = posição da linha.", path)
        out.insert(0, "point_id", np.arange(len(out)))
    for j in range(j_item + 1, j_fim):
        out[cabecalho[j]] = _numerico_se_possivel(dados[j]).to_numpy()
    out = _restore_sem_esquema(out)
    if "replication_id" not in out.columns:
        logger.warning("%s: sem coluna replication_id — ordem de aparição no point_id.", path)
        out.insert(1, "replication_id", out.groupby("point_id").cumcount().astype("int64"))
    return out


def _numerico_se_possivel(serie: pd.Series) -> pd.Series:
    """header=None faz o pandas ler cada coluna como object (o cabeçalho
    está misturado com os números). Converte para número quando NADA se
    perde; senão devolve como veio, para extract_results marcar `invalida`."""
    convertida = pd.to_numeric(serie, errors="coerce")
    if int(convertida.isna().sum()) == int(serie.isna().sum()):
        return convertida
    return serie
