"""
Fase 5 — SHAP: contribuição de cada variável para cada previsão.

O explicador depende da arquitetura (monografia, "Camada de
Explicabilidade"):

    TreeSHAP    RandomForest, ExtraTrees, GradientBoosting, XGBoost —
                cálculo exato e rápido, aproveitando a estrutura de árvore
    KernelSHAP  todo o resto (GP, MLP, baselines) — aproximação amostral,
                com fundo resumido por k-means para caber no orçamento

ESCALA: o SHAP é calculado sobre o modelo como ele foi treinado, então X
está na escala transformada [0, 1] e os valores SHAP estão na escala
transformada de y. A ORDEM das importâncias não depende disso — e é a
ordem que se compara com a teoria na Fase 6. Para magnitudes em unidade
do KPI, usar PDP/ICE (pdp_ice.py), que já respondem na escala original.

COLUNAS ONE-HOT: um parâmetro categórico vira várias colunas de X.
importance_by_parameter() soma as contribuições de volta por parâmetro —
SHAP é aditivo, então a soma é a contribuição do parâmetro.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ARVORES = ("RandomForestRegressor", "ExtraTreesRegressor",
            "GradientBoostingRegressor", "XGBRegressor")


@dataclass
class ShapResult:
    values: pd.DataFrame        # uma linha por ponto explicado, uma coluna por feature
    base_value: float           # previsão média de referência (escala transformada)
    method: str                 # "tree" ou "kernel"


def explainer_kind(model) -> str:
    """"tree" para as famílias de árvore, "kernel" para as demais."""
    return "tree" if type(model).__name__ in _ARVORES else "kernel"


def compute_shap(
    model,
    X_explain: np.ndarray,
    feature_names: list[str],
    *,
    X_background: np.ndarray | None = None,
    method: str = "auto",
    n_background: int = 50,
    nsamples: int | str = "auto",
    seed: int = 0,
) -> ShapResult:
    """Valores SHAP de `model` nos pontos de `X_explain` (já transformados).

    X_background: referência do KernelSHAP (em geral o X de treino); é
    resumido em `n_background` centróides de k-means. Ignorado no TreeSHAP,
    que usa a própria estrutura das árvores.
    """
    import shap

    X_explain = np.asarray(X_explain, dtype="float64")
    tipo = explainer_kind(model) if method == "auto" else method

    if tipo == "tree":
        explicador = shap.TreeExplainer(model)
        valores = explicador.shap_values(X_explain)
        base = explicador.expected_value
    elif tipo == "kernel":
        if X_background is None:
            raise ValueError("KernelSHAP precisa de X_background (ex.: o X de treino).")
        # réplicas repetem o X: o fundo é o conjunto de pontos distintos
        fundo = np.unique(np.asarray(X_background, dtype="float64"), axis=0)
        if len(fundo) > n_background:
            fundo = shap.kmeans(fundo, n_background)
        explicador = shap.KernelExplainer(model.predict, fundo, seed=seed)
        valores = explicador.shap_values(X_explain, nsamples=nsamples, silent=True)
        base = explicador.expected_value
    else:
        raise ValueError(f"method desconhecido: {method!r} (use 'auto', 'tree' ou 'kernel').")

    base = float(np.ravel(base)[0])
    return ShapResult(pd.DataFrame(np.asarray(valores), columns=feature_names), base, tipo)


def shap_importance(result: ShapResult) -> pd.DataFrame:
    """Importância global = média de |SHAP| por feature, em ordem decrescente,
    com a fração do total (soma 1)."""
    media = result.values.abs().mean().sort_values(ascending=False)
    total = media.sum()
    return pd.DataFrame({
        "feature": media.index,
        "mean_abs_shap": media.to_numpy(),
        "fracao": media.to_numpy() / total if total > 0 else np.nan,
        "rank": np.arange(1, len(media) + 1),
    }).reset_index(drop=True)


def importance_by_parameter(result: ShapResult, param_specs) -> pd.DataFrame:
    """Importância global por PARÂMETRO do YAML, somando as colunas one-hot.

    Para cada ponto, soma os SHAP das colunas do mesmo parâmetro e só então
    tira a média do valor absoluto (somar as médias de |SHAP| superestimaria
    a contribuição quando as colunas se compensam).
    """
    por_parametro = {}
    for spec in param_specs:
        if spec.name in result.values.columns:
            cols = [spec.name]
        else:
            cols = [c for c in result.values.columns if c.startswith(f"{spec.name}_")]
        if cols:
            por_parametro[spec.name] = result.values[cols].sum(axis=1)
    agregado = ShapResult(pd.DataFrame(por_parametro), result.base_value, result.method)
    return shap_importance(agregado).rename(columns={"feature": "parametro"})
