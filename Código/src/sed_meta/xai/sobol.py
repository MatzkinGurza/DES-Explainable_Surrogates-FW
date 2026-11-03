"""
Fase 5 — índices de Sobol sobre o metamodelo.

Decomposição da variância do KPI entre os parâmetros, sobre TODO o espaço
declarado no YAML (e não só sobre os pontos do DoE, como o SHAP):

    S1  fração da variância explicada pelo parâmetro sozinho
    ST  fração total, incluindo todas as interações em que ele participa

ST bem maior que S1 = o parâmetro atua sobretudo em interação. ST perto de
zero = o parâmetro não influencia o KPI no espaço estudado.

Com o simulador isso custaria N·(d+2) execuções do ProModel; com o
metamodelo, custa N·(d+2) previsões. A amostra é gerada no cubo unitário
pelo SALib (sequência de Sobol, estimadores de Saltelli/Jansen) e
decodificada para os limites do YAML pelo MESMO decodificador do DoE
(parameter_space._decode_unit_cube): inteiros e categorias recebem a mesma
discretização equiprovável.
"""

import numpy as np
import pandas as pd

from sed_meta.doe.parameter_space import _decode_unit_cube


def sobol_indices(
    surrogate,
    param_specs,
    *,
    n_base: int = 1024,
    seed: int = 0,
) -> pd.DataFrame:
    """S1 e ST (com intervalo de confiança de 95%) de cada parâmetro.

    surrogate : OriginalScaleSurrogate (ou chamável df -> y)
    n_base    : N da amostra de Saltelli; potência de 2. Custo = N·(d+2)
                previsões.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    if n_base < 2 or n_base & (n_base - 1):
        raise ValueError(f"n_base={n_base}: use uma potência de 2 (balanceamento da sequência de Sobol).")

    nomes = [s.name for s in param_specs]
    problema = {"num_vars": len(nomes), "names": nomes, "bounds": [[0.0, 1.0]] * len(nomes)}
    cubo = sobol_sample.sample(problema, n_base, calc_second_order=False, seed=seed)
    y = np.asarray(surrogate(_decode_unit_cube(cubo, param_specs)), dtype="float64")

    res = sobol_analyze.analyze(problema, y, calc_second_order=False, seed=seed)
    return pd.DataFrame({
        "parametro": nomes,
        "S1": res["S1"],
        "S1_conf": res["S1_conf"],
        "ST": res["ST"],
        "ST_conf": res["ST_conf"],
    }).sort_values("ST", ascending=False).reset_index(drop=True)
