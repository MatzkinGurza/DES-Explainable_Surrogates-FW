"""Limiares e utilitários compartilhados pelos testes da Fase 3.

Separado do conftest.py de propósito: fixtures o pytest injeta sozinho, mas
constantes e funções precisam ser importadas, e um conftest não é
importável por nome quando o diretório de testes é um pacote.
"""

import numpy as np

#: Limiar frouxo de acerto. O piso real do dataset é o ruído injetado
#: (~0.15 em unidades de desvio de y); 0.45 dá folga larga para as famílias
#: mais enviesadas — árvores sobre superfície suave, sobretudo — sem deixar
#: passar um modelo que não aprendeu (o dummy fica em ~1.0).
RMSE_MAXIMO = 0.45

#: Abaixo disto o dataset estaria fácil demais e o teste (c) — "o dummy é
#: pior que todo mundo" — deixaria de discriminar qualquer coisa.
RMSE_MINIMO_DO_DUMMY = 0.90


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def grid_reduzido(grid: dict, por_chave: int = 1) -> dict:
    """Encolhe um grid mantendo TODAS as chaves.

    Os grids sugeridos são grandes de propósito (o GRID_XGBOOST tem quase
    mil combinações). Rodá-los inteiros num teste unitário não caberia no
    tempo, e não é o que o teste (d) quer verificar: o que está em jogo ali
    é se cada CHAVE do grid corresponde a um parâmetro real do estimador e
    se o GridSearchCV consegue percorrer aquilo sem erro de assinatura.
    Manter todas as chaves e cortar os valores preserva exatamente isso.
    """
    return {chave: valores[:por_chave] for chave, valores in grid.items()}
