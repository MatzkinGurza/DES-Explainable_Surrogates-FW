"""Testes da família de árvores (Fase 3)."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from sed_meta.metamodels.baseline import build_dummy_mean
from sed_meta.metamodels.tree_family import (
    GRID_EXTRA_TREES,
    GRID_GRADIENT_BOOSTING,
    GRID_RANDOM_FOREST,
    GRID_XGBOOST,
    build_extra_trees,
    build_gradient_boosting,
    build_random_forest,
    build_xgboost,
    fit_with_sample_weight,
)

from tests.metamodels.helpers import (
    RMSE_MAXIMO,
    RMSE_MINIMO_DO_DUMMY,
    grid_reduzido,
    rmse,
)

VARIANTES = [
    ("random_forest", build_random_forest, GRID_RANDOM_FOREST),
    ("extra_trees", build_extra_trees, GRID_EXTRA_TREES),
    ("gradient_boosting", build_gradient_boosting, GRID_GRADIENT_BOOSTING),
    ("xgboost", build_xgboost, GRID_XGBOOST),
]

#: As duas subfamílias, para os testes que dependem da distinção
#: bagging/boosting registrada no docstring de tree_family.
BAGGING = ["random_forest", "extra_trees"]
BOOSTING = ["gradient_boosting", "xgboost"]


@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_fit_predict_e_acerto(dados, nome, construtor, _grid):
    """(a) fit/predict rodam; (b) o erro fica abaixo do limiar frouxo."""
    X_train, X_test, y_train, y_test, _ = dados

    modelo = construtor()
    modelo.fit(X_train, y_train)
    previsto = modelo.predict(X_test)

    assert previsto.shape == y_test.shape
    assert np.isfinite(previsto).all()
    assert rmse(y_test, previsto) < RMSE_MAXIMO, f"{nome} não aprendeu a relação"


def test_dummy_e_o_pior_de_todos(dados):
    """(c) sanidade do próprio teste."""
    X_train, X_test, y_train, y_test, _ = dados

    erro_dummy = rmse(
        y_test, build_dummy_mean().fit(X_train, y_train).predict(X_test)
    )
    assert erro_dummy > RMSE_MINIMO_DO_DUMMY

    for nome, construtor, _ in VARIANTES:
        erro = rmse(y_test, construtor().fit(X_train, y_train).predict(X_test))
        assert erro < erro_dummy, f"{nome} não ganhou de chutar a média"


@pytest.mark.parametrize("nome,construtor,grid", VARIANTES)
def test_grid_search_roda(dados, nome, construtor, grid):
    """(d) GridSearchCV percorre o grid sugerido sem erro de assinatura."""
    X_train, _X_test, y_train, _y_test, _ = dados

    busca = GridSearchCV(construtor(), grid_reduzido(grid), cv=3, error_score="raise")
    busca.fit(X_train, y_train)
    assert np.isfinite(busca.best_score_)


@pytest.mark.parametrize("nome,construtor,grid", VARIANTES)
def test_chaves_do_grid_sao_parametros_reais(nome, construtor, grid):
    """Complemento do (d): valida TODAS as chaves do grid completo."""
    estimador = construtor()
    for chave, valores in grid.items():
        estimador.set_params(**{chave: valores[0]})


# ---------------------------------------------------------------------
# O cuidado específico da família: n_estimators nos grids
# ---------------------------------------------------------------------

@pytest.mark.parametrize("nome,_construtor,grid", VARIANTES)
def test_bagging_nao_tuna_n_estimators_mas_boosting_sim(nome, _construtor, grid):
    """Trava a decisão registrada no docstring de tree_family.

    Bagging (RF/ExtraTrees): n_estimators FORA do grid — Breiman mostra que
    não sobreajusta com mais árvores, então buscar nesse eixo gasta
    orçamento por uma resposta já conhecida.

    Boosting (GB/XGBoost): n_estimators DENTRO do grid — ali é o número de
    passos de correção, e passar do ponto sobreajusta de verdade.
    """
    if nome in BAGGING:
        assert "n_estimators" not in grid, (
            f"{nome} é bagging: n_estimators não deve entrar no grid"
        )
        assert "max_depth" in grid and "min_samples_leaf" in grid, (
            f"{nome}: o grid precisa varrer max_depth e min_samples_leaf"
        )
    else:
        assert "n_estimators" in grid, (
            f"{nome} é boosting: n_estimators é eixo legítimo de busca"
        )
        assert "learning_rate" in grid, (
            f"{nome}: n_estimators é acoplado ao learning_rate"
        )


@pytest.mark.parametrize("nome,construtor", [(n, c) for n, c, _ in VARIANTES])
def test_n_estimators_e_configuravel_mesmo_fora_do_grid(nome, construtor):
    """Ficar fora do grid é decisão sobre a BUSCA, não sobre a API: o
    parâmetro continua ajustável por quem quiser mudá-lo à mão."""
    modelo = construtor(n_estimators=17)
    assert modelo.get_params()["n_estimators"] == 17


# ---------------------------------------------------------------------
# sample_weight
# ---------------------------------------------------------------------

@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_sample_weight_e_repassado(recorder, dados, nome, construtor, _grid):
    """Toda esta família aceita peso: o repasse tem de mudar o ajuste, e
    NÃO pode disparar o aviso de "peso ignorado" — se disparar, houve
    regressão em _destino_do_peso e o peso sumiria do ponto de vista de
    quem só olha o resultado."""
    X_train, X_test, y_train, _y_test, w_train = dados

    com_peso = construtor()
    fit_with_sample_weight(com_peso, X_train, y_train, sample_weight=w_train)
    sem_peso = construtor().fit(X_train, y_train)

    assert not recorder.avisos_de_peso_ignorado(), (
        f"{nome}: fit_with_sample_weight avisou que ignorou o peso"
    )
    assert not np.allclose(com_peso.predict(X_test), sem_peso.predict(X_test)), (
        f"{nome}: previsões idênticas com e sem peso — o sample_weight não chegou"
    )
