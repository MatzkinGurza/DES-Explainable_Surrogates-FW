"""
PDP e ICE, na escala original dos parâmetros e do indicador.

    ICE  uma curva por ponto: fixa as demais variáveis daquele ponto e varre
         a variável de interesse pela grade
    PDP  a média das curvas ICE: o efeito médio da variável

O PDP é uma média e pode esconder interação; as curvas ICE mostram se o
efeito é homogêneo (curvas paralelas) ou depende do resto do ponto (curvas
que se cruzam ou abrem em leque).

A grade vem dos limites DECLARADOS do parâmetro no YAML: float -> n_grid
pontos de low a high; integer -> todos os inteiros (ou n_grid deles);
categórico/ordinal -> níveis. As previsões passam por
OriginalScaleSurrogate, que reaplica o pré-processamento do treino.
"""

import numpy as np
import pandas as pd


def parameter_grid(spec, n_grid: int = 20) -> list:
    """Grade de valores de um parâmetro, a partir do spec."""
    if spec.type == "float":
        return list(np.linspace(spec.low, spec.high, n_grid))
    if spec.type == "integer":
        inteiros = np.arange(int(spec.low), int(spec.high) + 1)
        if len(inteiros) > n_grid:
            inteiros = np.unique(np.round(np.linspace(spec.low, spec.high, n_grid)).astype(int))
        return [int(v) for v in inteiros]
    if spec.type in ("categorical", "ordinal"):
        return list(spec.levels)
    if spec.type == "binary":
        return [0, 1] if spec.ordered else ["0", "1"]
    raise ValueError(f"Tipo de parâmetro desconhecido: {spec.type!r}")


def ice_curves(
    surrogate,
    df: pd.DataFrame,
    spec,
    *,
    n_grid: int = 20,
    max_pontos: int | None = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Curvas ICE de um parâmetro.

    surrogate : OriginalScaleSurrogate (ou qualquer chamável df -> y)
    df        : pontos de referência, com as colunas de parâmetro na escala
                do YAML (em geral os pontos do DoE, um por point_id)
    max_pontos: subamostra de df para limitar o custo (None = todos)

    Devolve formato longo: linha (índice do ponto em df), valor, predicao.
    """
    # réplicas repetem os parâmetros: uma curva por ponto distinto
    colunas = getattr(surrogate, "param_names", list(df.columns))
    base = df[colunas].drop_duplicates()
    if max_pontos is not None and len(base) > max_pontos:
        base = base.sample(max_pontos, random_state=seed)
    base = base.reset_index(drop=True)

    grade = parameter_grid(spec, n_grid)
    blocos = []
    for valor in grade:
        variado = base.copy()
        variado[spec.name] = valor
        blocos.append(pd.DataFrame({"linha": base.index, "valor": [valor] * len(base),
                                    "predicao": surrogate(variado)}))
    return pd.concat(blocos, ignore_index=True)


def pdp_from_ice(ice: pd.DataFrame) -> pd.DataFrame:
    """PDP = média das curvas ICE em cada valor da grade, com dispersão
    (desvio e faixa 10–90%) das curvas."""
    g = ice.groupby("valor", sort=False)["predicao"]
    return pd.DataFrame({
        "media": g.mean(),
        "desvio": g.std(ddof=1),
        "p10": g.quantile(0.10),
        "p90": g.quantile(0.90),
    }).reset_index()
