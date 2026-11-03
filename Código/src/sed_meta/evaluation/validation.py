"""
Fase 4 — esquemas de validação de UM metamodelo para UM KPI.

Posição no pipeline:

    preprocessing.PreprocessingPipeline  -> X_train/X_test, y_train/y_test
    metamodels.<família>.build_<x>()     -> estimador não ajustado
        -> ESTE MÓDULO                   -> métricas por fold, hold-out,
                                            curva de aprendizado, sementes
            -> diagnostics.diagnose()    -> gatilhos de revisão

Quatro funções, uma por pergunta da monografia (seção "Testes e Plano de
Avaliação Estatística"):

    cross_validate_grouped()  o modelo generaliza? (treino x validação por
                              fold, e a variância entre folds)
    holdout_evaluation()      qual o desempenho final, num teste que nunca
                              foi visto? (métrica que vai para a aba eval)
    learning_curve_grouped()  erro e tempo de treino em função do volume de
                              dados (as avaliações "qualitativas" da
                              monografia)
    seed_stability()          o treino converge do mesmo jeito com sementes
                              diferentes?

AGRUPAMENTO POR PONTO, SEMPRE: com dados por replicação, réplicas do mesmo
ponto têm o mesmo X. Fold ou subamostra que separasse réplicas de um ponto
avaliaria o modelo num X que ele já viu (mesmo motivo do split agrupado de
preprocessing.split()). Por isso toda função que reparte dados recebe
`groups` (o point_id de cada linha) e reparte PONTOS, não linhas.

Todas recebem o estimador NÃO ajustado e trabalham sobre clones: nada aqui
altera o objeto de quem chama. O ajuste passa por fit_with_sample_weight(),
que repassa o peso só a quem aceita e avisa quando não aceita.

`inverse`: função que leva y da escala transformada para a original —
em geral `lambda y: pipe.inverse_transform_y(y, kpi)`. Com ela as métricas
saem na escala original (as que se reportam); a escala transformada fica
em `r2_t` (é nela que o modelo de fato ajusta). Sem ela, tudo fica na
escala transformada.
"""

import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from sed_meta.evaluation.metrics import regression_metrics
from sed_meta.metamodels.baseline import fit_with_sample_weight


def _metricas(y_t, y_pred_t, inverse) -> dict[str, float]:
    """Métricas na escala original (se houver inverse) + R² na transformada."""
    m = regression_metrics(inverse(y_t), inverse(y_pred_t)) if inverse else regression_metrics(y_t, y_pred_t)
    m["r2_t"] = regression_metrics(y_t, y_pred_t)["r2"]
    return m


def _ajustar(estimator, X, y, sample_weight):
    modelo = clone(estimator)
    t0 = time.perf_counter()
    fit_with_sample_weight(modelo, X, y, sample_weight=sample_weight)
    return modelo, time.perf_counter() - t0


def _fatia(peso, idx):
    return None if peso is None else np.asarray(peso)[idx]


# ---------------------------------------------------------------------
# 1. Validação cruzada agrupada
# ---------------------------------------------------------------------

def cross_validate_grouped(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups,
    *,
    n_splits: int = 5,
    sample_weight=None,
    inverse: Callable | None = None,
) -> pd.DataFrame:
    """K-fold agrupado por ponto (GroupKFold), com métricas de treino E de
    validação em cada fold.

    Devolve uma linha por (fold, conjunto): colunas fold, conjunto
    ("treino"/"validacao"), n_pontos, n_linhas, fit_time_s e as métricas.
    As duas leituras que importam saem daqui: a distância treino x validação
    (sobreajuste) e o desvio da validação entre folds (instabilidade) — ver
    diagnostics.summarize_cv().
    """
    X, y, groups = np.asarray(X), np.asarray(y, dtype="float64"), np.asarray(groups)
    n_grupos = len(np.unique(groups))
    if n_grupos < n_splits:
        raise ValueError(f"{n_grupos} pontos distintos para {n_splits} folds.")

    linhas = []
    for fold, (idx_tr, idx_va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        modelo, t_fit = _ajustar(estimator, X[idx_tr], y[idx_tr], _fatia(sample_weight, idx_tr))
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


# ---------------------------------------------------------------------
# 2. Hold-out (a medida final, não enviesada pelo treino)
# ---------------------------------------------------------------------

def holdout_evaluation(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    sample_weight=None,
    inverse: Callable | None = None,
) -> dict:
    """Ajusta no treino inteiro e mede no teste hold-out.

    O teste tem de ter saído de preprocessing.split() (agrupado por ponto)
    e não pode ter sido usado na escolha de hiperparâmetros — para isso
    existe a validação cruzada, dentro do treino.

    Devolve: modelo ajustado, métricas de treino e de teste, tempo de ajuste
    e de previsão (s), e as previsões de teste nas duas escalas.
    """
    modelo, t_fit = _ajustar(estimator, X_train, y_train, sample_weight)
    t0 = time.perf_counter()
    pred_teste_t = modelo.predict(X_test)
    t_pred = time.perf_counter() - t0
    return {
        "modelo": modelo,
        "treino": _metricas(np.asarray(y_train), modelo.predict(X_train), inverse),
        "teste": _metricas(np.asarray(y_test), pred_teste_t, inverse),
        "fit_time_s": t_fit,
        "predict_time_s": t_pred,
        "y_pred_teste_t": pred_teste_t,
        "y_pred_teste": inverse(pred_teste_t) if inverse else pred_teste_t,
    }


# ---------------------------------------------------------------------
# 3. Curva de aprendizado (erro e tempo x volume de dados)
# ---------------------------------------------------------------------

def learning_curve_grouped(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 0,
    sample_weight=None,
    inverse: Callable | None = None,
) -> pd.DataFrame:
    """Treina em frações crescentes dos PONTOS de treino e mede sempre no
    mesmo teste.

    As frações são aninhadas (os pontos de 20% estão nos de 40%, e assim por
    diante), então a curva mede o efeito do volume, não o de trocar de
    amostra. Uma linha por fração: n_pontos, n_linhas, fit_time_s, métricas
    de treino (prefixo treino_) e de teste (prefixo teste_).

    Leitura: teste ainda caindo na última fração = mais pontos de DoE ainda
    ajudam; teste estável = o volume não é o gargalo. fit_time_s mostra onde
    o custo de uma família cresce rápido (ex.: GP, custo cúbico).
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train, dtype="float64")
    groups_train = np.asarray(groups_train)
    pontos = np.random.default_rng(seed).permutation(np.unique(groups_train))

    linhas = []
    for f in sorted(fractions):
        if not 0 < f <= 1:
            raise ValueError(f"fração {f} fora de (0, 1].")
        escolhidos = pontos[: max(2, int(round(f * len(pontos))))]
        idx = np.flatnonzero(np.isin(groups_train, escolhidos))
        modelo, t_fit = _ajustar(estimator, X_train[idx], y_train[idx], _fatia(sample_weight, idx))
        treino = _metricas(y_train[idx], modelo.predict(X_train[idx]), inverse)
        teste = _metricas(np.asarray(y_test), modelo.predict(X_test), inverse)
        linhas.append({
            "fracao": f,
            "n_pontos": len(escolhidos),
            "n_linhas": len(idx),
            "fit_time_s": t_fit,
            **{f"treino_{k}": v for k, v in treino.items()},
            **{f"teste_{k}": v for k, v in teste.items()},
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------
# 4. Estabilidade entre sementes
# ---------------------------------------------------------------------

def seed_stability(
    build_estimator: Callable[[int], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    sample_weight=None,
    inverse: Callable | None = None,
) -> pd.DataFrame:
    """Mesmo modelo, mesmos dados, sementes diferentes.

    `build_estimator(seed)` devolve o estimador configurado com aquela
    semente (ex.: `lambda s: build_random_forest().set_params(random_state=s)`). Só faz
    sentido para modelos com aleatoriedade no treino (RF, boosting, MLP,
    otimização de hiperparâmetros do GP); para os determinísticos a
    variação sai zero, o que também é uma informação.

    Uma linha por semente, com métricas de teste e fit_time_s.
    """
    linhas = []
    for s in seeds:
        modelo, t_fit = _ajustar(build_estimator(s), X_train, y_train, sample_weight)
        linhas.append({"seed": s, "fit_time_s": t_fit,
                       **_metricas(np.asarray(y_test), modelo.predict(X_test), inverse)})
    return pd.DataFrame(linhas)
