"""
Métricas de fidelidade preditiva da monografia:

    R²    variância explicada; qualidade geral do ajuste
    RMSE  penaliza erros grandes; sensível a pontos atípicos
    MAE   erro típico; robusto a pontos atípicos

A razão RMSE/MAE acusa erro concentrado numa região do espaço: vale 1 para
erros todos do mesmo tamanho e cresce quanto mais o erro se concentra.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """R², RMSE, MAE, MSE e RMSE/MAE de um vetor de previsões.

    NaN em y_true ou y_pred levanta erro em vez de ser descartado: uma
    previsão faltando é defeito a corrigir antes, não linha a ignorar.
    """
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} e y_pred {y_pred.shape} com tamanhos diferentes.")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN em y_true ou y_pred.")

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "rmse_mae": rmse / mae if mae > 0 else float("nan"),
    }
