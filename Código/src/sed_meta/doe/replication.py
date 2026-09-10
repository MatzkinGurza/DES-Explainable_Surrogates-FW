"""
Fase 1 (continuação) — Replicações.

Responsabilidade deste módulo: decidir QUANTAS vezes rodar a simulação
em cada ponto do design (política fixa, sem lógica adaptativa), montar
o plano de execução correspondente, e depois — dado o resultado bruto
dessas execuções — agregar em um dataset de treino e calcular o teto
de ruído por KPI.

Este módulo é puro e sem I/O: ele não roda a simulação. Quem roda é o
ProModel, a partir de uma planilha Excel ("cockpit") em que UMA LINHA =
UMA EXECUÇÃO (um ponto do DoE x uma replicação) — exatamente a
granularidade das N x R linhas produzidas aqui por
build_execution_plan(). O caminho completo:

    build_execution_plan()  -> sim_interface/excel_bridge.py (cockpit .xlsx)
    ... ProModel preenche as colunas de KPI da planilha ...
    excel_bridge.extract_results() -> excel_bridge.select_complete_runs()
    -> `raw` -> as funções abaixo

É por não ter I/O que as funções de agregação podem ser testadas com
DataFrame sintético, sem depender de planilha nenhuma.

REGRA DO PROJETO — NADA É DELETADO SILENCIOSAMENTE: nenhuma linha
desaparece em nenhum ponto do pipeline. Quem lê o cockpit
(excel_bridge.extract_results) devolve TODAS as execuções, marcadas por
status; a seleção das completas é feita à parte, de forma explícita e
visível, por excel_bridge.select_complete_runs(). As funções de
agregação abaixo assumem receber apenas execuções completas — mas essa
seleção é responsabilidade declarada de quem chama, e elas avisam
(logging WARNING) se receberem KPI vazio, em vez de descartar por conta
própria.
"""

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 1. Política de replicação (fixa, por decisão do projeto)
# ---------------------------------------------------------------------

@dataclass
class ReplicationPolicy:
    """Política fixa: todo ponto do design roda o mesmo número de
    replicações. Sem lógica adaptativa por variância."""
    n_replications: int = 5
    seed_base: int = 0


# ---------------------------------------------------------------------
# 2. Expansão do design em plano de execução (pura, sem I/O)
# ---------------------------------------------------------------------

def build_execution_plan(design: pd.DataFrame, policy: ReplicationPolicy) -> pd.DataFrame:
    """Expande o design (N pontos únicos, saída do DoE) em um plano de
    execução de N x n_replications linhas.

    Cada linha do plano tem:
      - point_id: identifica de qual linha do design essa réplica veio
      - replication_id: 0..n_replications-1, dentro daquele ponto
      - seed: determinística, derivada de point_id e replication_id,
        para reprodutibilidade (mesma seed_base -> mesmo plano sempre)
      - as colunas de parâmetros, repetidas em cada réplica do ponto

    Este DataFrame é o que vira o cockpit .xlsx que o ProModel percorre
    linha a linha — aqui ainda não se roda nada.

    Granularidade: estas N x R linhas são exatamente a granularidade
    "uma linha por execução" consumida pelo cockpit Excel em
    sed_meta/sim_interface/excel_bridge.py. build_cockpit_table() recebe
    este DataFrame como está (mantendo point_id, replication_id, seed e
    os parâmetros como bloco de ENTRADA, nesta ordem) e só acrescenta à
    direita as colunas de KPI vazias que o ProModel vai preencher.

    PRESERVAÇÃO DE DTYPE: a expansão é vetorizada (repeat sobre o índice)
    de propósito. Montar o plano linha a linha (iterrows + row.to_dict
    + pd.DataFrame(rows)) faz o pandas re-inferir os tipos do zero: uma
    coluna `category` — inclusive a ORDENADA que representa um fator
    ordinal (ex.: baixa < media < alta, ver
    parameter_space.ParameterSpec) — chegaria aqui como string solta,
    perdendo os níveis e a ordem antes mesmo do cockpit. Com
    `design.loc[design.index.repeat(R)]` o dtype vem preservado de
    origem, sem precisar ser restaurado depois.
    """
    design = design.reset_index(drop=True)
    n_points, n_rep = len(design), policy.n_replications

    plan = design.loc[design.index.repeat(n_rep)].reset_index(drop=True)

    # point_id crescente, replication_id 0..R-1 dentro de cada ponto.
    point_id = np.repeat(np.arange(n_points), n_rep)
    replication_id = np.tile(np.arange(n_rep), n_points)
    seed = policy.seed_base + point_id * n_rep + replication_id

    # insert() na posição 0, em ordem inversa, deixa o bloco de
    # bookkeeping à esquerda: point_id | replication_id | seed | params...
    plan.insert(0, "seed", seed)
    plan.insert(0, "replication_id", replication_id)
    plan.insert(0, "point_id", point_id)
    return plan


# ---------------------------------------------------------------------
# 3. Agregação das replicações em dataset de treino (pura, sem I/O)
# ---------------------------------------------------------------------

def _avisar_kpi_vazio(raw: pd.DataFrame, kpi_cols: list[str], origem: str) -> None:
    """Sinaliza execução incompleta que chegou até a agregação.

    Não filtra nada: pela regra do projeto, o descarte tem que ser
    explícito e feito por quem chama (excel_bridge.select_complete_runs).
    Aqui só garantimos que o caso não passe em silêncio.
    """
    presentes = [k for k in kpi_cols if k in raw.columns]
    vazios = {k: int(raw[k].isna().sum()) for k in presentes}
    vazios = {k: n for k, n in vazios.items() if n > 0}
    if not vazios:
        return

    linhas_afetadas = int(raw[presentes].isna().any(axis=1).sum())
    logger.warning(
        "%s recebeu %d linha(s) com KPI vazio (%s). Elas NÃO foram descartadas: "
        "o groupby ignora o NaN, então o n desses pontos fica menor. "
        "Rode excel_bridge.select_complete_runs() antes se a intenção era "
        "agregar só execuções completas.",
        origem, linhas_afetadas, vazios,
    )


def aggregate_replications(
    raw: pd.DataFrame,
    param_cols: list[str],
    kpi_cols: list[str],
) -> pd.DataFrame:
    """Agrega resultados brutos (uma linha por replicação já executada,
    com colunas point_id + param_cols + kpi_cols) em um dataset de
    treino: uma linha por ponto, com média, erro padrão da média (SEM)
    e peso amostral (inverso da variância) de cada KPI.

    O `sample_weight` resultante é o gancho para metamodelos que aceitam
    peso por amostra (ex.: hetGP, ou regressão ponderada) — pontos mais
    ruidosos recebem peso menor no ajuste.

    De onde vem `raw`: do cockpit .xlsx lido por
    sim_interface/excel_bridge.py — extract_results() (devolve TODAS as
    execuções, marcadas por status) seguido de select_complete_runs() (a
    seleção explícita das completas). Em teste, `raw` pode ser um
    DataFrame sintético: o contrato de colunas (point_id + param_cols +
    kpi_cols) é o mesmo, e colunas extras como `status_execucao` são
    ignoradas.

    Espera-se aqui SÓ execuções completas. Se vier KPI vazio, esta função
    não descarta a linha por conta própria (regra do projeto: nada some
    em silêncio) — ela avisa em WARNING e segue, deixando o groupby
    ignorar o NaN, o que reduz o n daquele ponto. Rodar
    select_complete_runs() antes é o que evita esse caso.
    """
    _avisar_kpi_vazio(raw, kpi_cols, "aggregate_replications")

    params = raw.groupby("point_id")[param_cols].first()

    per_kpi_frames = []
    for kpi in kpi_cols:
        grouped = raw.groupby("point_id")[kpi]
        frame = pd.DataFrame({
            f"{kpi}_mean": grouped.mean(),
            f"{kpi}_std": grouped.std(ddof=1),
            f"{kpi}_n": grouped.count(),
        })
        frame[f"{kpi}_sem"] = frame[f"{kpi}_std"] / np.sqrt(frame[f"{kpi}_n"])
        # variância zero (todas as réplicas idênticas) -> peso indefinido;
        # marcamos como NaN em vez de dividir por zero silenciosamente
        var = frame[f"{kpi}_std"] ** 2
        frame[f"{kpi}_sample_weight"] = np.where(var > 0, 1.0 / var, np.nan)
        per_kpi_frames.append(frame)

    dataset = params.join(per_kpi_frames[0])
    for frame in per_kpi_frames[1:]:
        dataset = dataset.join(frame)

    return dataset.reset_index()


# ---------------------------------------------------------------------
# 4. Teto de ruído por KPI (pura, sem I/O)
# ---------------------------------------------------------------------

def compute_noise_ceiling(raw: pd.DataFrame, kpi_cols: list[str]) -> dict[str, float]:
    """Teto de ruído por KPI, via decomposição tipo ANOVA one-way:

        variância_total = variância_entre_pontos + variância_intra_ponto

    noise_ceiling = variância_entre_pontos / variância_total

    Interpretação: é a fração da variância observada que é sinal real
    (diferença genuína entre pontos do espaço de parâmetros), versus
    ruído estocástico do próprio simulador (diferença entre réplicas
    do MESMO ponto). Nenhum metamodelo pode ter R² acima deste valor
    de forma legítima — se tiver, é sinal de overfitting ao ruído.

    noise_ceiling perto de 1 -> simulador pouco ruidoso, quase todo R²
    faltante é culpa do metamodelo.
    noise_ceiling baixo -> simulador muito ruidoso naquele KPI; um R²
    modesto pode já estar no limite do que é fisicamente possível.

    De onde vem `raw`: igual a aggregate_replications() — do cockpit
    .xlsx (extract_results() seguido de select_complete_runs(), em
    sim_interface/excel_bridge.py). Vale um cuidado de leitura: como só
    entram execuções COMPLETAS, um cockpit parcialmente preenchido
    produz um teto de ruído provisório, estimado sobre menos réplicas do
    que o plano previa.

    Se vier KPI vazio, avisa em WARNING e segue sem descartar linha
    (regra do projeto: nada some em silêncio).
    """
    _avisar_kpi_vazio(raw, kpi_cols, "compute_noise_ceiling")

    ceilings = {}
    for kpi in kpi_cols:
        grouped = raw.groupby("point_id")[kpi]
        within_var_per_point = grouped.var(ddof=1).dropna()
        n_per_point = grouped.count()

        dof = (n_per_point.loc[within_var_per_point.index] - 1)
        if dof.sum() == 0:
            # menos de 2 réplicas por ponto em todo o dataset -> não dá
            # para estimar variância intra-ponto
            ceilings[kpi] = float("nan")
            continue

        pooled_within_var = np.average(within_var_per_point, weights=dof)
        total_var = raw[kpi].var(ddof=1)

        if total_var <= 0:
            ceilings[kpi] = float("nan")
            continue

        between_var = max(total_var - pooled_within_var, 0.0)
        ceilings[kpi] = between_var / total_var

    return ceilings