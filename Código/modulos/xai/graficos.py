"""
Gráficos de explicabilidade no formato das figuras da monografia.

    grafico_shap()      dispersão dos valores SHAP dos parâmetros mais
                        importantes, em segundos, colorida pelo valor da entrada
    grafico_pdp_ice()   PDP (linha azul, faixa entre os percentis 10 e 90) e
                        uma amostra de curvas ICE (cinza) para alguns parâmetros
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modulos.xai.pdp_ice import ice_curves, pdp_from_ice

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def grafico_shap(shap_s: pd.DataFrame, pontos: pd.DataFrame, specs: dict, kpi: str, parametros: list[str]):
    """Uma linha por parâmetro (o mais importante no topo); cada ponto é uma
    previsão. `shap_s` são os valores SHAP já em segundos, uma coluna por
    parâmetro; `pontos` são as entradas explicadas, na escala do YAML."""
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for i, p in enumerate(reversed(parametros)):
        x = pontos[p].to_numpy(dtype=float)
        lo, hi = specs[p].low, specs[p].high
        cor = (x - lo) / (hi - lo) if hi > lo else x * 0
        y = i + np.random.default_rng(i).uniform(-0.25, 0.25, len(x))
        sc = ax.scatter(shap_s[p], y, c=cor, cmap="coolwarm", vmin=0, vmax=1, s=12, edgecolor="none")
    ax.set_yticks(range(len(parametros)))
    ax.set_yticklabels(list(reversed(parametros)))
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"Contribuição SHAP para {kpi} (s)")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["mín.", "máx."])
    cb.set_label("Valor da entrada")
    fig.tight_layout()
    return fig


def grafico_pdp_ice(surrogate, train_df: pd.DataFrame, specs: dict, kpi: str, parametros: list[str],
                    unidades: dict, n_grid: int = 15, n_curvas_ice: int = 40, seed: int = 0):
    """Um painel por parâmetro, com eixo y compartilhado (previsão em segundos)."""
    fig, axs = plt.subplots(1, len(parametros), figsize=(6.5, 2.4), sharey=True, squeeze=False)
    for ax, p in zip(axs[0], parametros):
        ice = ice_curves(surrogate, train_df, specs[p], n_grid=n_grid, max_pontos=200, seed=seed)
        amostra = ice["linha"].drop_duplicates().sample(n_curvas_ice, random_state=seed)
        for _, g in ice[ice["linha"].isin(amostra)].groupby("linha"):
            ax.plot(g["valor"], g["predicao"], color="0.75", lw=0.5)
        pdp = pdp_from_ice(ice)
        ax.fill_between(pdp["valor"], pdp["p10"], pdp["p90"], color="tab:blue", alpha=0.18, lw=0)
        ax.plot(pdp["valor"], pdp["media"], color="tab:blue", lw=2)
        ax.set_xlabel(f"{p} ({unidades.get(p, '')})", fontsize=8.5)
    axs[0][0].set_ylabel(f"{kpi} previsto (s)", fontsize=8.5)
    fig.tight_layout()
    return fig
