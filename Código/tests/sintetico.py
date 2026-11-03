"""Dados sintéticos com resposta CONHECIDA, para testar Fases 4 e 5 antes de
haver resultado do ProModel.

    y = 8·a + 4·b² + 2·a·b + 1·[regra == "SPT"] + 0·c + ruído N(0, 0,5)

Ordem de importância esperada: a > b > regra > c (c não tem efeito nenhum).
Dados por replicação (K réplicas por ponto), como os do cockpit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sed_meta.doe.parameter_space import ParameterSpec, generate_design
from sed_meta.doe.replication import ReplicationPolicy, build_execution_plan
from sed_meta.preprocessing import PreprocessingPipeline, TargetTransformSpec

SPECS = [
    ParameterSpec("a", "float", low=0.0, high=1.0),
    ParameterSpec("b", "float", low=0.0, high=1.0),
    ParameterSpec("c", "integer", low=1, high=3),
    ParameterSpec("regra", "categorical", levels=["FIFO", "SPT"]),
]
KPI = "y"
RUIDO = 0.5


def resposta(df: pd.DataFrame) -> np.ndarray:
    return (8 * df["a"] + 4 * df["b"] ** 2 + 2 * df["a"] * df["b"]
            + 1.0 * (df["regra"].astype(str) == "SPT")).to_numpy(dtype="float64")


@dataclass
class Caso:
    raw: pd.DataFrame
    pipe: PreprocessingPipeline
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray

    def inverse(self, y_t):
        return self.pipe.inverse_transform_y(y_t, KPI)


def caso(n_pontos: int = 80, n_rep: int = 3, seed: int = 0) -> Caso:
    design = generate_design(SPECS, n_points=n_pontos, seed=seed)
    raw = build_execution_plan(design, ReplicationPolicy(n_replications=n_rep))
    raw[KPI] = resposta(raw) + np.random.default_rng(seed).normal(0, RUIDO, len(raw))

    pipe = PreprocessingPipeline(SPECS, [TargetTransformSpec(KPI, "standard")])
    train_df, test_df = pipe.split(raw, test_size=0.25, seed=seed)
    pipe.fit(train_df)
    return Caso(raw, pipe, train_df, test_df,
                pipe.transform_X(train_df), pipe.transform_y(train_df, KPI),
                pipe.transform_X(test_df), pipe.transform_y(test_df, KPI),
                train_df["point_id"].to_numpy())
