"""
Fase 2 — interface com o simulador, via cockpit Excel.

A troca de dados com o ProModel é feita por uma planilha .xlsx
("cockpit") no formato:

    uma linha = uma execução completa (um ponto do DoE x uma replicação)

    [ ---- ENTRADA (esquerda) ---- ] [ ---- SAÍDA (direita) ---- ]
    point_id | replication_id | seed | <params...> | <KPIs...>
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
# São entrada do cockpit (o ProModel precisa da seed), mas NÃO são
# parâmetros do modelo — ver nota em extract_results().
ID_COLS = ("point_id", "replication_id", "seed")

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
           colunas point_id, replication_id, seed + colunas de parâmetro.
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
    seed e parâmetros inteiros, que o Excel devolve como 0.0, 1.0, 2.0).
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
    replication_id, seed, colunas de parâmetro, colunas de KPI e
    status_execucao. As colunas de KPI vêm convertidas para numérico —
    valor não numérico vira NaN na coluna, mas o valor original continua
    no .xlsx e é reportado no log, e a linha fica marcada `invalida`.

    Cuidado ao derivar `param_cols` para aggregate_replications(): use as
    colunas do design, ou exclua explicitamente ID_COLS + kpi_names +
    STATUS_COL. point_id/replication_id/seed são bookkeeping, não fatores
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

    point_id é obrigatório (validado antes); replication_id e seed podem
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
