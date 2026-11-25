"""
Replicações: expande os N pontos do DoE em N x R execuções.

Política fixa: todo ponto roda o mesmo número de replicações. O plano de
execução tem uma linha por execução (um ponto x uma replicação), que é a
granularidade de uma linha do cockpit do ProModel. Cada linha recebe
point_id, replication_id e unique_id.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Política de replicação (fixa, por decisão do projeto)
# ---------------------------------------------------------------------

@dataclass
class ReplicationPolicy:
    """Política fixa: todo ponto do design roda o mesmo número de
    replicações. Sem lógica adaptativa por variância."""
    n_replications: int = 5
    id_base: int = 0


# ---------------------------------------------------------------------
# 2. Expansão do design em plano de execução (pura, sem I/O)
# ---------------------------------------------------------------------

def build_execution_plan(design: pd.DataFrame, policy: ReplicationPolicy) -> pd.DataFrame:
    """Expande o design (N pontos únicos, saída do DoE) em um plano de
    execução de N x n_replications linhas.

    Cada linha do plano tem:
      - point_id: identifica de qual linha do design essa réplica veio
      - replication_id: 0..n_replications-1, dentro daquele ponto
      - unique_id: id único da execução, determinístico
        (id_base + point_id * n_replications + replication_id)
      - as colunas de parâmetros, repetidas em cada réplica do ponto

    Este DataFrame é o que vira o cockpit .xlsx que o ProModel percorre
    linha a linha — aqui ainda não se roda nada.

    Granularidade: estas N x R linhas são exatamente a granularidade
    "uma linha por execução" consumida pelo cockpit Excel em
    sim_interface/excel_bridge.py. build_cockpit_table() recebe
    este DataFrame como está (mantendo point_id, replication_id, unique_id e
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
    unique_id = policy.id_base + point_id * n_rep + replication_id

    # insert() na posição 0, em ordem inversa, deixa o bloco de
    # bookkeeping à esquerda: point_id | replication_id | unique_id | params...
    plan.insert(0, "unique_id", unique_id)
    plan.insert(0, "replication_id", replication_id)
    plan.insert(0, "point_id", point_id)
    return plan
