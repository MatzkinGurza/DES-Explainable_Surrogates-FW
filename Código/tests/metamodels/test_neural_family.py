"""Testes da família neural (Fase 3)."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from sed_meta.metamodels.baseline import build_dummy_mean
from sed_meta.metamodels.neural_family import (
    GRID_MLP,
    build_mlp,
    fit_with_sample_weight,
)

from tests.metamodels.helpers import (
    RMSE_MAXIMO,
    RMSE_MINIMO_DO_DUMMY,
    grid_reduzido,
    rmse,
)


def test_fit_predict_e_acerto(dados):
    """(a) fit/predict rodam; (b) o erro fica abaixo do limiar frouxo."""
    X_train, X_test, y_train, y_test, _ = dados

    modelo = build_mlp()
    modelo.fit(X_train, y_train)
    previsto = modelo.predict(X_test)

    assert previsto.shape == y_test.shape
    assert np.isfinite(previsto).all()
    assert rmse(y_test, previsto) < RMSE_MAXIMO


def test_dummy_e_o_pior(dados):
    """(c) sanidade do próprio teste."""
    X_train, X_test, y_train, y_test, _ = dados

    erro_dummy = rmse(
        y_test, build_dummy_mean().fit(X_train, y_train).predict(X_test)
    )
    erro_mlp = rmse(y_test, build_mlp().fit(X_train, y_train).predict(X_test))

    assert erro_dummy > RMSE_MINIMO_DO_DUMMY
    assert erro_mlp < erro_dummy


def test_grid_search_roda(dados):
    """(d) GridSearchCV percorre o grid sugerido sem erro de assinatura."""
    X_train, _X_test, y_train, _y_test, _ = dados

    busca = GridSearchCV(
        build_mlp(), grid_reduzido(GRID_MLP), cv=3, error_score="raise"
    )
    busca.fit(X_train, y_train)
    assert np.isfinite(busca.best_score_)


def test_chaves_do_grid_sao_parametros_reais():
    """Complemento do (d): valida TODAS as chaves do grid completo."""
    estimador = build_mlp()
    for chave, valores in GRID_MLP.items():
        estimador.set_params(**{chave: valores[0]})


# ---------------------------------------------------------------------
# As decisões registradas no docstring da família
# ---------------------------------------------------------------------

def test_solver_padrao_e_lbfgs():
    """A escolha que mais importa nesta família: lbfgs (segunda ordem,
    lote inteiro) em vez de adam (estocástico, minilote), porque o regime
    é de poucas centenas de pontos."""
    assert build_mlp().get_params()["solver"] == "lbfgs"


def test_lbfgs_ganha_de_adam_neste_regime(dados):
    """Sustenta a decisão acima com medida, em vez de deixá-la como
    afirmação de docstring."""
    X_train, X_test, y_train, y_test, _ = dados

    erro_lbfgs = rmse(
        y_test, build_mlp(solver="lbfgs").fit(X_train, y_train).predict(X_test)
    )
    erro_adam = rmse(
        y_test, build_mlp(solver="adam").fit(X_train, y_train).predict(X_test)
    )
    assert erro_lbfgs < erro_adam


def test_grid_nao_traz_parametros_de_minilote():
    """Com solver="lbfgs" o sklearn IGNORA learning_rate_init, batch_size e
    early_stopping. Pô-los no grid multiplicaria o custo da busca sem mudar
    resultado nenhum — desperdício silencioso, do tipo que ninguém percebe
    porque a busca "funciona"."""
    ignorados = {"learning_rate_init", "batch_size", "early_stopping", "learning_rate"}
    assert not (ignorados & set(GRID_MLP)), (
        f"grid traz parâmetro ignorado por lbfgs: {ignorados & set(GRID_MLP)}"
    )


def test_regularizacao_padrao_e_mais_forte_que_a_do_sklearn():
    """alpha=1e-1 contra o default 1e-4 do sklearn: com resposta ruidosa e
    poucos pontos, a rede tem capacidade de sobra para memorizar o ruído."""
    assert build_mlp().get_params()["alpha"] >= 1e-1


def test_defaults_convergem_de_verdade(dados):
    """Trava a correção mais sutil desta família.

    Um max_iter apertado funciona como EARLY STOPPING DISFARÇADO: segura o
    sobreajuste sem estar declarado em lugar nenhum, e portanto sem o
    GridSearchCV poder controlá-lo enquanto varre alpha. Se este teste
    falhar, o modelo voltou a depender do teto de iteração como
    regularizador implícito — a correção é subir alpha, não subir max_iter.
    """
    X_train, _X_test, y_train, _y_test, _ = dados

    with warnings.catch_warnings(record=True) as registro:
        warnings.simplefilter("always")
        modelo = build_mlp().fit(X_train, y_train)

    convergencia = [
        a for a in registro if issubclass(a.category, ConvergenceWarning)
    ]
    assert not convergencia, (
        f"o lbfgs não convergiu em {modelo.get_params()['max_iter']} iterações "
        f"(parou em {modelo.n_iter_})"
    )
    assert modelo.n_iter_ < modelo.get_params()["max_iter"], (
        "n_iter_ encostou no teto: a regularização está vindo do limite de "
        "iteração, não do alpha"
    )


def test_alpha_fraco_sobreajusta_e_alpha_forte_nao(dados):
    """A medição que justifica o default: abaixo de 1e-1 a rede melhora no
    treino e PIORA no teste — assinatura de estar ajustando ruído."""
    X_train, X_test, y_train, y_test, _ = dados

    fraco = build_mlp(alpha=1e-4).fit(X_train, y_train)
    forte = build_mlp(alpha=3e-1).fit(X_train, y_train)

    assert rmse(y_train, fraco.predict(X_train)) < rmse(y_train, forte.predict(X_train))
    assert rmse(y_test, forte.predict(X_test)) < rmse(y_test, fraco.predict(X_test))


def test_alpha_controla_o_sobreajuste(dados):
    """Confirma que o eixo principal do grid faz o que se espera dele: mais
    penalização, menos aderência ao treino."""
    X_train, _X_test, y_train, _y_test, _ = dados

    frouxo = build_mlp(alpha=1e-6).fit(X_train, y_train)
    apertado = build_mlp(alpha=10.0).fit(X_train, y_train)

    assert rmse(y_train, frouxo.predict(X_train)) < rmse(
        y_train, apertado.predict(X_train)
    )


def test_nao_ha_pytorch_na_familia():
    """Decisão de escopo registrada no docstring do módulo: PyTorch só
    entra se o MLP se mostrar insuficiente na prática, o que ainda não foi
    testado. Este teste é o lembrete executável disso."""
    import sed_meta.metamodels.neural_family as familia

    fonte = familia.__file__
    with open(fonte, encoding="utf-8") as arquivo:
        texto = arquivo.read()

    assert "import torch" not in texto
    assert "NÃO ADICIONE PYTORCH" in texto, (
        "a decisão de escopo saiu do docstring — se foi revertida de "
        "propósito, atualize este teste junto"
    )


# ---------------------------------------------------------------------
# sample_weight
# ---------------------------------------------------------------------

def test_sample_weight_e_repassado(recorder, dados):
    """MLPRegressor aceita peso (sklearn >= 1.7), então o peso por ponto do
    aggregate_replications() chega ao treino como nas árvores."""
    X_train, X_test, y_train, _y_test, w_train = dados

    com_peso = build_mlp()
    fit_with_sample_weight(com_peso, X_train, y_train, sample_weight=w_train)
    sem_peso = build_mlp().fit(X_train, y_train)

    assert not recorder.avisos_de_peso_ignorado(), (
        "o MLP aceita sample_weight; o aviso indica regressão no roteamento"
    )
    assert not np.allclose(com_peso.predict(X_test), sem_peso.predict(X_test))
