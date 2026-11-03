"""
Fase 4 — métricas de fidelidade preditiva.

As três métricas da monografia (Tabela "Métricas de fidelidade preditiva"),
mais o MSE, que a aba eval do results-by-test também registra:

    R²    variância explicada; qualidade geral do ajuste
    RMSE  penaliza erros grandes; sensível a pontos atípicos
    MAE   erro típico; robusto a pontos atípicos
    MSE   RMSE ao quadrado (só para o registro)

A razão RMSE/MAE vai junto porque é ela que acusa o padrão "MAE baixo com
RMSE alto" (erro concentrado numa região do espaço): para erros todos do
mesmo tamanho ela vale 1, e cresce quanto mais o erro se concentra.

TETO DE RUÍDO: replication.compute_noise_ceiling() dá, por KPI, a fração da
variância que é sinal (o resto é ruído do simulador). Nenhum metamodelo
pode ter R² legitimamente acima dele quando avaliado contra RÉPLICAS
individuais. gap_to_noise_ceiling() expressa o R² nessa régua.
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


def gap_to_noise_ceiling(r2: float, noise_ceiling: float) -> dict[str, float]:
    """Posição do R² em relação ao teto de ruído do KPI.

    fracao_do_teto = R² / teto: 1 = o modelo extrai todo o sinal que existe;
    acima de 1 = R² maior que o fisicamente possível contra réplicas, o que
    indica vazamento (ex.: split não agrupado) ou sobreajuste ao ruído.
    """
    if not np.isfinite(noise_ceiling) or noise_ceiling <= 0:
        return {"teto": float(noise_ceiling), "gap": float("nan"), "fracao_do_teto": float("nan")}
    return {
        "teto": float(noise_ceiling),
        "gap": float(noise_ceiling - r2),
        "fracao_do_teto": float(r2 / noise_ceiling),
    }
