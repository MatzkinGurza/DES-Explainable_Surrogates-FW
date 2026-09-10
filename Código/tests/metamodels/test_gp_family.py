"""Testes da família de processos gaussianos (Fase 3).

Além dos quatro itens exigidos, esta suíte carrega o peso de validar os
WRAPPERS: HetGPRegressor e SMTKrigingRegressor existem só para traduzir
APIs, e é justamente aí que a compatibilidade com sklearn quebra em
silêncio (clone() que perde parâmetro, predict que devolve (n,1) em vez de
(n,), get_params que não enxerga um argumento).
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from sed_meta.metamodels.baseline import build_dummy_mean
from sed_meta.metamodels.gp_family import (
    GRID_HETGP,
    GRID_SKLEARN_GP,
    GRID_SMT_KRIGING,
    HetGPRegressor,
    SMTKrigingRegressor,
    build_hetgp,
    build_sklearn_gp,
    build_smt_kriging,
    fit_with_sample_weight,
)

from tests.metamodels.helpers import (
    RMSE_MAXIMO,
    RMSE_MINIMO_DO_DUMMY,
    grid_reduzido,
    rmse,
)

VARIANTES = [
    ("sklearn_gp", build_sklearn_gp, GRID_SKLEARN_GP),
    ("hetgp", build_hetgp, GRID_HETGP),
    ("smt_kriging", build_smt_kriging, GRID_SMT_KRIGING),
]


@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_fit_predict_e_acerto(dados, nome, construtor, _grid):
    """(a) fit/predict rodam; (b) o erro fica abaixo do limiar frouxo."""
    X_train, X_test, y_train, y_test, _ = dados

    modelo = construtor()
    modelo.fit(X_train, y_train)
    previsto = modelo.predict(X_test)

    assert previsto.shape == y_test.shape, (
        f"{nome}: predict devolveu {previsto.shape}, esperado {y_test.shape} — "
        "o sklearn espera (n,) em regressão de alvo único"
    )
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
# O que distingue esta família: variância preditiva nativa
# ---------------------------------------------------------------------

@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_variancia_preditiva_nativa(dados, nome, construtor, _grid):
    """A razão de ser da família: todas devolvem desvio-padrão preditivo
    junto da média, sem conformal prediction por cima."""
    X_train, X_test, y_train, _y_test, _ = dados

    modelo = construtor().fit(X_train, y_train)
    media, desvio = modelo.predict(X_test, return_std=True)

    assert media.shape == desvio.shape == (len(X_test),)
    assert np.isfinite(desvio).all(), f"{nome}: desvio com NaN/inf"
    assert (desvio >= 0).all(), f"{nome}: desvio negativo"
    assert desvio.max() > 0, f"{nome}: desvio identicamente zero"


def test_hetgp_soma_nugget_na_variancia_preditiva(dados):
    """A variância preditiva completa é sd2 + nugs (documentação do
    hetgpy). Usar só sd2 produziria intervalos otimistas de forma
    sistemática — bug silencioso, porque o número continua saindo."""
    X_train, X_test, y_train, _y_test, _ = dados

    modelo = build_hetgp().fit(X_train, y_train)
    _, desvio = modelo.predict(X_test, return_std=True)

    bruto = modelo.model_.predict(x=X_test)
    so_sd2 = np.sqrt(np.clip(np.asarray(bruto["sd2"]).ravel(), 0, None))

    assert (desvio >= so_sd2 - 1e-12).all()
    assert desvio.mean() > so_sd2.mean(), (
        "o nugget não entrou na variância preditiva"
    )


def test_kriging_com_eval_noise_reporta_mais_incerteza(dados):
    """eval_noise=True é desvio deliberado do default do SMT.

    O efeito confiável NÃO é no RMSE de treino: o otimizador às vezes
    encontra sozinho uma solução suave mesmo sem estimar ruído, e neste
    dataset os dois ajustes erram parecido no treino. O efeito robusto é na
    INCERTEZA REPORTADA — sem estimar o ruído, o KRG fica confiante demais,
    e é a variância preditiva que dá metade do valor desta família.
    """
    X_train, _X_test, y_train, _y_test, _ = dados

    com_ruido = build_smt_kriging(eval_noise=True).fit(X_train, y_train)
    sem_ruido = build_smt_kriging(eval_noise=False).fit(X_train, y_train)

    _, sd_com = com_ruido.predict(X_train, return_std=True)
    _, sd_sem = sem_ruido.predict(X_train, return_std=True)

    assert sd_com.mean() > sd_sem.mean(), (
        "com eval_noise=True o KRG devia reportar MAIS incerteza; "
        f"medido {sd_com.mean():.5f} contra {sd_sem.mean():.5f}"
    )
    # e continua suavizando: o erro de treino fica na ordem do ruído
    # injetado (~0.145 em unidades de desvio), não em zero
    assert rmse(y_train, com_ruido.predict(X_train)) > 0.05


def test_eval_noise_fica_fora_do_grid():
    """eval_noise é premissa do problema (a resposta É ruidosa), não
    hiperparâmetro. No grid, um fold sortudo poderia escolher a versão
    confiante demais."""
    assert "eval_noise" not in GRID_SMT_KRIGING


def test_sklearn_gp_estima_o_nivel_de_ruido(dados):
    """O WhiteKernel não é decorativo: sem ele o GP interpola. O nível
    estimado deve ficar na ordem do ruído real do dataset."""
    X_train, _X_test, y_train, _y_test, _ = dados

    modelo = build_sklearn_gp().fit(X_train, y_train)
    ruido = modelo.kernel_.k2.noise_level

    # y foi padronizado; o ruído injetado (0.1) sobre desvio de y (~0.65)
    # dá ~0.15 em desvio, ou ~0.024 em variância. Faixa larga de propósito.
    assert 1e-3 < ruido < 1e-1, f"noise_level estimado fora do esperado: {ruido}"


# ---------------------------------------------------------------------
# Wrappers: compatibilidade com sklearn
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "wrapper,params",
    [
        (HetGPRegressor, {"model": "homGP", "covtype": "Matern5_2", "maxit": 50}),
        (SMTKrigingRegressor, {"corr": "matern52", "poly": "linear", "n_start": 3}),
    ],
)
def test_wrapper_sobrevive_a_clone(wrapper, params):
    """clone() é o que o GridSearchCV usa para cada combinação. Ele
    reconstrói o objeto só a partir de get_params(), então qualquer
    parâmetro que o __init__ transforme em vez de armazenar some aqui."""
    original = wrapper(**params)
    copia = clone(original)

    assert copia.get_params() == original.get_params()
    for chave, valor in params.items():
        assert getattr(copia, chave) == valor


@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_cross_val_score_roda(dados, nome, construtor, _grid):
    """cross_val_score exige fit/predict/score e clone — o pacote completo
    da API. É o teste mais direto de que o wrapper é um estimador de fato."""
    X_train, _X_test, y_train, _y_test, _ = dados

    notas = cross_val_score(construtor(), X_train, y_train, cv=3)
    assert len(notas) == 3
    assert np.isfinite(notas).all()
    assert notas.mean() > 0.5, f"{nome}: R² médio em CV baixo demais ({notas.mean()})"


@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_atributos_de_ajuste_seguem_a_convencao(dados, nome, construtor, _grid):
    X_train, _X_test, y_train, _y_test, _ = dados
    modelo = construtor().fit(X_train, y_train)
    assert modelo.n_features_in_ == X_train.shape[1]


def test_hetgp_recusa_model_desconhecido(dados):
    X_train, _X_test, y_train, _y_test, _ = dados
    with pytest.raises(ValueError, match="desconhecido"):
        HetGPRegressor(model="nao_existe").fit(X_train, y_train)


@pytest.mark.parametrize("wrapper", [HetGPRegressor, SMTKrigingRegressor])
def test_wrapper_recusa_x_e_y_de_tamanhos_diferentes(dados, wrapper):
    X_train, _X_test, y_train, _y_test, _ = dados
    with pytest.raises(ValueError):
        wrapper().fit(X_train, y_train[:-5])


# ---------------------------------------------------------------------
# sample_weight: esta é a família que NÃO usa peso
# ---------------------------------------------------------------------

@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_peso_e_ignorado_com_aviso(recorder, dados, nome, construtor, _grid):
    """Nenhuma das três consome sample_weight. O contrato do módulo é
    ignorar COM AVISO — nunca em silêncio, e nunca inventando ponderação
    própria para compensar."""
    X_train, X_test, y_train, _y_test, w_train = dados

    modelo = construtor()
    fit_with_sample_weight(modelo, X_train, y_train, sample_weight=w_train)

    assert recorder.avisos_de_peso_ignorado(), (
        f"{nome}: o peso foi descartado sem aviso nenhum"
    )
    # e o ajuste acontece mesmo assim, sem peso
    assert np.isfinite(modelo.predict(X_test)).all()


@pytest.mark.parametrize("nome,construtor,_grid", VARIANTES)
def test_sem_peso_nao_avisa_nada(recorder, dados, nome, construtor, _grid):
    """O aviso é sobre peso DESCARTADO. Quem não passou peso não deve ser
    incomodado — senão o aviso vira ruído e para de ser lido."""
    X_train, _X_test, y_train, _y_test, _ = dados

    fit_with_sample_weight(construtor(), X_train, y_train)
    assert not recorder.avisos_de_peso_ignorado()
