"""Testes da família baseline (Fase 3)."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from sed_meta.metamodels.baseline import (
    GRID_DUMMY_MEAN,
    GRID_LINEAR,
    GRID_POLYNOMIAL,
    GRID_POLYNOMIAL_RIDGE,
    GRID_RIDGE,
    build_dummy_mean,
    build_linear,
    build_polynomial,
    build_polynomial_ridge,
    build_ridge,
    fit_with_sample_weight,
)

from tests.metamodels.helpers import (
    RMSE_MAXIMO,
    RMSE_MINIMO_DO_DUMMY,
    grid_reduzido,
    rmse,
)

#: (nome, construtor, grid) — o dummy fica de fora porque ele é o piso
#: contra o qual os outros são medidos, não um modelo que deva acertar.
VARIANTES = [
    ("linear", build_linear, GRID_LINEAR),
    ("polynomial", build_polynomial, GRID_POLYNOMIAL),
    ("ridge", build_ridge, GRID_RIDGE),
    ("polynomial_ridge", build_polynomial_ridge, GRID_POLYNOMIAL_RIDGE),
]


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
    """(c) sanidade do próprio teste: se o dummy não fosse o pior, o dataset
    não teria sinal e os limiares acima não estariam medindo nada."""
    X_train, X_test, y_train, y_test, _ = dados

    dummy = build_dummy_mean().fit(X_train, y_train)
    erro_dummy = rmse(y_test, dummy.predict(X_test))

    assert erro_dummy > RMSE_MINIMO_DO_DUMMY, (
        "o dummy acertou demais — o dataset sintético perdeu o sinal"
    )
    for nome, construtor, _ in VARIANTES:
        erro = rmse(y_test, construtor().fit(X_train, y_train).predict(X_test))
        assert erro < erro_dummy, f"{nome} não ganhou de chutar a média"


def test_dummy_preve_a_media_do_treino(dados):
    """O piso é literalmente a média de y_train, e o R² dele no PRÓPRIO
    conjunto de ajuste é 0 por construção — é o que faz dele um piso."""
    X_train, X_test, y_train, _y_test, _ = dados

    dummy = build_dummy_mean().fit(X_train, y_train)
    previsto = dummy.predict(X_test)

    assert np.allclose(previsto, y_train.mean())
    assert dummy.score(X_train, y_train) == pytest.approx(0.0, abs=1e-12)


def test_polinomial_recupera_a_relacao_quadratica(dados):
    """A relação sintética é exatamente de 2ª ordem, então o polinomial
    quadrático deve chegar perto do piso de ruído — bem abaixo do linear."""
    X_train, X_test, y_train, y_test, _ = dados

    erro_linear = rmse(y_test, build_linear().fit(X_train, y_train).predict(X_test))
    erro_poly = rmse(
        y_test, build_polynomial(degree=2).fit(X_train, y_train).predict(X_test)
    )

    assert erro_poly < erro_linear
    assert erro_poly < 0.20, "o polinomial de grau 2 devia chegar perto do ruído"


@pytest.mark.parametrize("nome,construtor,grid", VARIANTES)
def test_grid_search_roda(dados, nome, construtor, grid):
    """(d) GridSearchCV percorre o grid sugerido sem erro de assinatura."""
    X_train, _X_test, y_train, _y_test, _ = dados

    busca = GridSearchCV(construtor(), grid_reduzido(grid), cv=3, error_score="raise")
    busca.fit(X_train, y_train)
    assert np.isfinite(busca.best_score_)


@pytest.mark.parametrize(
    "nome,construtor,grid",
    VARIANTES + [("dummy_mean", build_dummy_mean, GRID_DUMMY_MEAN)],
)
def test_chaves_do_grid_sao_parametros_reais(nome, construtor, grid):
    """Complemento do (d): valida TODAS as chaves do grid completo, não só
    as do recorte que o GridSearchCV percorre no teste acima. set_params
    levanta se o nome não existir no estimador."""
    estimador = construtor()
    for chave, valores in grid.items():
        estimador.set_params(**{chave: valores[0]})


def test_sample_weight_chega_ao_passo_final_do_pipeline(dados):
    """O roteamento `reg__sample_weight` é a razão de _destino_do_peso
    existir: sem ele os baselines polinomiais — que são Pipeline —
    perderiam o peso em silêncio."""
    X_train, _X_test, y_train, _y_test, w_train = dados

    modelo = build_polynomial_ridge()
    fit_with_sample_weight(modelo, X_train, y_train, sample_weight=w_train)
    com_peso = modelo.named_steps["reg"].coef_.copy()

    sem_peso = build_polynomial_ridge().fit(X_train, y_train).named_steps["reg"].coef_

    assert not np.allclose(com_peso, sem_peso), (
        "coeficientes idênticos com e sem peso: o sample_weight não chegou "
        "ao passo final do Pipeline"
    )


def test_sample_weight_repassado_aos_estimadores_simples(dados):
    """Fora de Pipeline o caminho é o direto, e o dummy é o caso mais
    legível: a previsão vira a média PONDERADA de y_train."""
    X_train, X_test, y_train, _y_test, w_train = dados

    modelo = build_dummy_mean()
    fit_with_sample_weight(modelo, X_train, y_train, sample_weight=w_train)

    assert modelo.predict(X_test)[0] == pytest.approx(np.average(y_train, weights=w_train))
