"""
Fase 5 — PDP e ICE, na escala original dos parâmetros e do KPI.

    ICE  uma curva por ponto: fixa as demais variáveis daquele ponto e varre
         a variável de interesse pela grade
    PDP  a média das curvas ICE: o efeito médio da variável

Por que os dois (monografia, Tabela "Técnicas de explicabilidade"): o PDP é
uma média e pode esconder interação — duas subpopulações com efeitos
opostos dão um PDP plano. As curvas ICE mostram se o efeito é homogêneo
(curvas paralelas) ou depende do resto do ponto (curvas que se cruzam ou
abrem em leque). ice_heterogeneity() resume isso num número.

Implementação própria, e não sklearn.inspection.partial_dependence, porque
a grade aqui é na escala do YAML (minutos, capacidades inteiras) e a
resposta na escala do KPI: tudo passa por OriginalScaleSurrogate, que
reaplica o preprocessing do treino.

A GRADE vem dos limites DECLARADOS do parâmetro (ParameterSpec), não dos
observados: float -> n_grid pontos de low a high; integer -> todos os
inteiros (ou n_grid deles, se forem muitos); categórico/ordinal -> níveis.
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


def centered_ice(ice: pd.DataFrame) -> pd.DataFrame:
    """ICE centrado (c-ICE): cada curva menos o seu valor no primeiro ponto
    da grade. Tira o nível de cada ponto e deixa só a forma do efeito."""
    primeiro = ice["valor"].iloc[0]
    ancora = ice.loc[ice["valor"] == primeiro].set_index("linha")["predicao"]
    out = ice.copy()
    out["predicao"] = out["predicao"] - out["linha"].map(ancora)
    return out


def ice_heterogeneity(ice: pd.DataFrame) -> float:
    """Quão pouco paralelas são as curvas ICE, na escala do KPI.

    = média, ao longo da grade, do desvio das curvas centradas, dividida
    pelo desvio de todas as previsões da ICE (a variabilidade do KPI que o
    modelo enxerga nesses pontos). Perto de 0: efeito homogêneo (sem
    interação com as demais variáveis). Valores altos: o efeito da variável
    depende do resto do ponto — o PDP sozinho engana.

    Dividir pela variabilidade total, e não pela amplitude do PDP, evita que
    um parâmetro de efeito médio quase nulo apareça com índice enorme.
    NaN se as previsões forem constantes.
    """
    c = centered_ice(ice)
    dispersao = c.groupby("valor", sort=False)["predicao"].std(ddof=1).mean()
    total = float(ice["predicao"].std(ddof=1))
    return float(dispersao / total) if total > 0 else float("nan")
