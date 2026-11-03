"""
Fase 3/4 — busca de hiperparâmetros com validação cruzada AGRUPADA.

Dentro de cada família, os hiperparâmetros são escolhidos por validação
cruzada no TREINO (monografia, "Metamodelos e Treinamento"); o teste
hold-out fica intocado para a medida final. Esta função é o GridSearchCV
do sklearn com uma única diferença que importa: os folds são GroupKFold por
point_id, para que réplicas do mesmo ponto não caiam em treino e validação
ao mesmo tempo (ver validation.py).

`n_iter`: None = grade completa (GridSearchCV); um inteiro = `n_iter`
combinações sorteadas da grade (RandomizedSearchCV). As grades de RF e MLP
têm ~60 combinações; com 5 folds isso são ~300 ajustes por família, então o
sorteio é o padrão do script.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, ParameterGrid, RandomizedSearchCV


def tune_grouped(
    estimator,
    grid: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups,
    *,
    n_splits: int = 5,
    n_iter: int | None = None,
    scoring: str = "r2",
    seed: int = 0,
    n_jobs: int | None = None,
) -> dict:
    """Busca agrupada; devolve o estimador ajustado no treino inteiro com os
    melhores hiperparâmetros.

    Devolve: estimador (best_estimator_), params, cv_score (média do
    `scoring` na validação, escala transformada), cv_score_std, n_candidatos,
    tempo_s e a tabela cv_results (uma linha por candidato).

    Grade vazia (dummy, linear) = nada a buscar: o estimador é avaliado só
    com a CV agrupada, para o score ficar comparável com o das outras famílias.
    """
    cv = GroupKFold(n_splits=n_splits)
    t0 = time.perf_counter()
    candidatos = len(ParameterGrid(grid)) if grid else 1

    if not grid:
        busca = GridSearchCV(clone(estimator), {}, cv=cv, scoring=scoring, n_jobs=n_jobs)
    elif n_iter is None or n_iter >= candidatos:
        busca = GridSearchCV(clone(estimator), grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    else:
        busca = RandomizedSearchCV(clone(estimator), grid, n_iter=n_iter, cv=cv, scoring=scoring,
                                   random_state=seed, n_jobs=n_jobs)
        candidatos = n_iter
    busca.fit(X, y, groups=groups)

    i = busca.best_index_
    return {
        "estimador": busca.best_estimator_,
        "params": busca.best_params_,
        "cv_score": float(busca.cv_results_["mean_test_score"][i]),
        "cv_score_std": float(busca.cv_results_["std_test_score"][i]),
        "n_candidatos": candidatos,
        "tempo_s": time.perf_counter() - t0,
        "cv_results": pd.DataFrame(busca.cv_results_),
    }
