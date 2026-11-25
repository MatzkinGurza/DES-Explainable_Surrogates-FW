"""
Validação cruzada agrupada por ponto do plano amostral.

As réplicas de um mesmo ponto têm as mesmas entradas. Uma partição que
separasse réplicas de um ponto avaliaria o modelo num X que ele já viu no
treino. Por isso as partições são de PONTOS (GroupKFold sobre o point_id),
não de linhas.

`inverse` leva y da escala transformada para a original (em geral
`lambda y: pipe.inverse_transform_y(y, kpi)`): com ela as métricas saem em
segundos, como na monografia; a escala transformada fica em `r2_t`.
"""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from modulos.evaluation.metrics import regression_metrics


def _metricas(y_t, y_pred_t, inverse) -> dict[str, float]:
    """Métricas na escala original (se houver inverse) + R² na transformada."""
    m = regression_metrics(inverse(y_t), inverse(y_pred_t)) if inverse else regression_metrics(y_t, y_pred_t)
    m["r2_t"] = regression_metrics(y_t, y_pred_t)["r2"]
    return m


def cross_validate_grouped(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups,
    *,
    n_splits: int = 5,
    inverse: Callable | None = None,
) -> pd.DataFrame:
    """K-fold agrupado por ponto, com métricas de treino E de validação.

    Recebe o estimador NÃO ajustado e treina um clone por partição. Devolve
    uma linha por (fold, conjunto): fold, conjunto ("treino"/"validacao"),
    n_pontos, n_linhas, fit_time_s e as métricas.
    """
    X, y, groups = np.asarray(X), np.asarray(y, dtype="float64"), np.asarray(groups)
    n_grupos = len(np.unique(groups))
    if n_grupos < n_splits:
        raise ValueError(f"{n_grupos} pontos distintos para {n_splits} folds.")

    linhas = []
    for fold, (idx_tr, idx_va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        modelo = clone(estimator)
        t0 = time.perf_counter()
        modelo.fit(X[idx_tr], y[idx_tr])
        t_fit = time.perf_counter() - t0
        for conjunto, idx in (("treino", idx_tr), ("validacao", idx_va)):
            linhas.append({
                "fold": fold,
                "conjunto": conjunto,
                "n_pontos": len(np.unique(groups[idx])),
                "n_linhas": len(idx),
                "fit_time_s": t_fit,
                **_metricas(y[idx], modelo.predict(X[idx]), inverse),
            })
    return pd.DataFrame(linhas)
