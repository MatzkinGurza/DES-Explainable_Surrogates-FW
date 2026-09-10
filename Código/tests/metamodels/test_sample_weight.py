"""Testes do contrato de sample_weight, atravessando as quatro famílias.

Este arquivo existe por causa de uma decisão de projeto com um custo: os
quatro arquivos de família são INDEPENDENTES entre si (nenhum importa o
outro), e o preço disso é que fit_with_sample_weight()/_destino_do_peso()
aparecem copiados em cada um. Cópia que ninguém vigia diverge — então aqui
as quatro são travadas como byte-idênticas.

O resto do arquivo verifica o contrato em si, uma vez só, em vez de repetir
o mesmo teste nas quatro suítes de família.
"""

import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sed_meta.metamodels import baseline, gp_family, neural_family, tree_family

FAMILIAS = [baseline, tree_family, gp_family, neural_family]

#: Delimitadores do bloco duplicado dentro de cada arquivo.
_INICIO = "def fit_with_sample_weight(estimator, X, y, sample_weight=None):"
_FIM = 'return "sample_weight" if has_fit_parameter(estimator, "sample_weight") else None'


def _bloco_duplicado(modulo) -> str:
    """Recorta o bloco de repasse de peso do arquivo-fonte da família."""
    with open(modulo.__file__, encoding="utf-8") as arquivo:
        texto = arquivo.read()

    inicio = texto.index(_INICIO)
    fim = texto.index(_FIM, inicio) + len(_FIM)
    return texto[inicio:fim]


def test_as_quatro_copias_sao_identicas():
    """Trava a duplicação deliberada. Se este teste falhar, ou alguém
    editou uma cópia e esqueceu as outras — corrija todas — ou a decisão de
    manter as famílias independentes foi revertida, e aí o bloco deve virar
    um módulo compartilhado e este teste sai junto."""
    blocos = {m.__name__.split(".")[-1]: _bloco_duplicado(m) for m in FAMILIAS}
    referencia = blocos["baseline"]

    divergentes = [nome for nome, bloco in blocos.items() if bloco != referencia]
    assert not divergentes, (
        f"o bloco de sample_weight divergiu em: {divergentes}. "
        "As quatro cópias precisam ser idênticas — ver a nota nos arquivos."
    )


@pytest.mark.parametrize("modulo", FAMILIAS, ids=lambda m: m.__name__.split(".")[-1])
def test_cada_familia_expoe_a_funcao(modulo):
    assert callable(modulo.fit_with_sample_weight)
    assert callable(modulo._destino_do_peso)


# ---------------------------------------------------------------------
# O contrato em si
# ---------------------------------------------------------------------

def test_estimador_simples_recebe_o_peso_direto():
    assert baseline._destino_do_peso(DummyRegressor()) == "sample_weight"
    assert baseline._destino_do_peso(LinearRegression()) == "sample_weight"


def test_pipeline_roteia_para_o_passo_final():
    """Pipeline.fit() não tem parâmetro `sample_weight` — has_fit_parameter
    devolve False para ela. Sem o roteamento explícito, todo baseline
    polinomial perderia o peso sem que nada avisasse."""
    from sklearn.utils.validation import has_fit_parameter

    pipe = Pipeline([("poly", PolynomialFeatures(2)), ("reg", LinearRegression())])

    assert has_fit_parameter(pipe, "sample_weight") is False
    assert baseline._destino_do_peso(pipe) == "reg__sample_weight"


def test_pipeline_com_passo_final_sem_peso_devolve_none():
    """Se o passo final não aceita peso, não há rota — e o caminho do aviso
    é o correto, não inventar `<passo>__sample_weight` e estourar no fit."""
    pipe = Pipeline([("scaler", StandardScaler()), ("gp", GaussianProcessRegressor())])
    assert baseline._destino_do_peso(pipe) is None


def test_nome_do_passo_final_e_respeitado():
    """O roteamento usa o nome real do passo, não um "reg" hard-coded."""
    pipe = Pipeline([("p", PolynomialFeatures(2)), ("meu_modelo", LinearRegression())])
    assert baseline._destino_do_peso(pipe) == "meu_modelo__sample_weight"


def test_quem_nao_aceita_peso_e_ajustado_assim_mesmo_com_aviso(dados):
    """O contrato tem duas metades e as duas importam: ignora o peso (não
    estoura) E avisa (não silencia)."""
    X_train, X_test, y_train, _y_test, w_train = dados
    modelo = GaussianProcessRegressor()

    with pytest.warns(UserWarning, match="não aceita sample_weight"):
        baseline.fit_with_sample_weight(modelo, X_train, y_train, sample_weight=w_train)

    assert np.isfinite(modelo.predict(X_test)).all()


def test_sem_peso_nao_avisa(dados):
    X_train, _X_test, y_train, _y_test, _ = dados

    with warnings.catch_warnings(record=True) as registro:
        warnings.simplefilter("always")
        baseline.fit_with_sample_weight(LinearRegression(), X_train, y_train)

    assert not [a for a in registro if "sample_weight" in str(a.message)]


def test_devolve_o_proprio_estimador(dados):
    """fit_with_sample_weight repassa o retorno de fit(), que por convenção
    do sklearn é o próprio estimador — para permitir encadear."""
    X_train, _X_test, y_train, _y_test, w_train = dados

    modelo = LinearRegression()
    devolvido = baseline.fit_with_sample_weight(
        modelo, X_train, y_train, sample_weight=w_train
    )
    assert devolvido is modelo
