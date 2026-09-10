"""Fixtures compartilhadas pelos testes da Fase 3.

O dataset sintético imita o que o preprocessing entrega: X escalado para
[0,1] (é o que transform_X() produz a partir dos limites do ParameterSpec)
e y padronizado (é o que transform_y() produz com method="standard"). Com y
padronizado, o RMSE sai em unidades de desvio-padrão, o que torna os
limiares dos testes legíveis:

    RMSE ~ 1.0  -> não aprendeu nada (é o que o DummyRegressor entrega)
    RMSE ~ 0.15 -> teto do dataset (o próprio ruído gaussiano injetado)

A relação é conhecida de propósito — y = 2*x1 + x2**2 + ruído — para que
"o modelo aprendeu" seja uma afirmação verificável e não um chute sobre
qual erro é aceitável.

Os limiares e utilitários ficam em helpers.py (ver o porquê lá).
"""

import warnings

import numpy as np
import pytest

SEED = 20240917


class _RegistroDeAvisos:
    """Captura os avisos emitidos no teste e sabe reconhecer o de peso
    ignorado, emitido por fit_with_sample_weight quando o estimador não
    aceita sample_weight."""

    def __init__(self, registro):
        self._registro = registro

    def avisos_de_peso_ignorado(self) -> list[str]:
        return [
            str(a.message)
            for a in self._registro
            if "sample_weight" in str(a.message)
            and ("não aceita" in str(a.message) or "não usa" in str(a.message)
                 or "ignorado" in str(a.message))
        ]


@pytest.fixture
def recorder():
    """Registra os avisos do teste, para afirmar tanto que um aviso
    esperado apareceu quanto que um aviso indevido NÃO apareceu."""
    with warnings.catch_warnings(record=True) as registro:
        warnings.simplefilter("always")
        yield _RegistroDeAvisos(registro)


@pytest.fixture(scope="session")
def dados():
    """(X_train, X_test, y_train, y_test, w_train) — relação conhecida.

    x3 entra INERTE de propósito: no DoE real há parâmetros que não afetam
    um dado KPI, e um metamodelo que quebra ou piora feio na presença de
    coluna irrelevante teria um problema que o teste precisa expor.

    O peso amostral imita a coluna f"{kpi}_sample_weight" de
    aggregate_replications() (1/variância por ponto): positivo, com
    dispersão moderada. Serve para checar que o repasse funciona, não para
    medir ganho de ponderação.
    """
    rng = np.random.default_rng(SEED)
    n, n_test = 200, 60
    total = n + n_test

    X = rng.uniform(0.0, 1.0, size=(total, 3))
    ruido = rng.normal(0.0, 0.1, size=total)
    y = 2.0 * X[:, 0] + X[:, 1] ** 2 + ruido

    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]

    # padronização com as estatísticas do TREINO — a mesma regra do
    # preprocessing (fit só no treino), para o teste não normalizar usando
    # o que ele mesmo vai avaliar
    media, desvio = y_train.mean(), y_train.std(ddof=0)
    y_train = (y_train - media) / desvio
    y_test = (y_test - media) / desvio

    w_train = rng.uniform(0.5, 2.0, size=n)
    return X_train, X_test, y_train, y_test, w_train
