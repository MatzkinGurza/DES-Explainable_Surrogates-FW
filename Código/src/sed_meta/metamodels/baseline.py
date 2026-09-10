"""
Fase 3 — família BASELINE: o piso de comparação.

Esta é a família que nunca deve ser pulada. Ela existe para responder, antes
de qualquer modelo caro, duas perguntas que decidem a leitura de todo o
resto da monografia:

  1. O metamodelo sofisticado ganha de chutar a média? Se não ganha, o
     problema não está no modelo — está nos dados, no DoE ou no ruído.
  2. A superfície de resposta é simples? Se uma regressão polinomial de 2ª
     ordem já resolve, ISSO É UM RESULTADO, não um fracasso: significa que
     a superfície de resposta do sistema simulado é suave e de baixa ordem,
     e essa é uma conclusão publicável sobre o SISTEMA — mais informativa,
     inclusive, do que um Random Forest opaco com o mesmo R². Não trate o
     sucesso do polinomial como sinal de que "faltou modelo melhor".

POSIÇÃO NO PIPELINE

    preprocessing.PreprocessingPipeline -> X_train/X_test, y_train/y_test
        -> ESTE MÓDULO (treina)         -> modelo ajustado
            -> evaluation (Fase 4)      -> R², RMSE, comparação com o teto
                                           de ruído de compute_noise_ceiling()

ESPAÇO TRANSFORMADO: os estimadores daqui treinam e preveem no espaço JÁ
TRANSFORMADO pelo preprocessing — X escalado para [0,1] pelos limites do
ParameterSpec, y padronizado (e possivelmente em log1p). NÃO se chama
inverse_transform_y aqui. Devolver previsão em minutos/peças é
responsabilidade da Fase 4, que tem o objeto de preprocessing em mãos.

UM MODELO POR KPI: X é compartilhado entre todos os KPIs, y não. O laço
"para cada KPI, treine um modelo" é de quem chama — este módulo só entrega
o estimador de um KPI por vez:

    for kpi in kpis:
        modelo = build_polynomial_ridge()
        fit_with_sample_weight(modelo, X_train, pipe.transform_y(train_df, kpi),
                               sample_weight=train_df[f"{kpi}_sample_weight"])

FORA DE ESCOPO: métricas (Fase 4), SHAP/PDP (Fase 5), laço por KPI e
controlador de ciclo.
"""

import warnings

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import has_fit_parameter


# ---------------------------------------------------------------------
# Repasse de sample_weight
# ---------------------------------------------------------------------
# NOTA SOBRE A DUPLICAÇÃO: este bloco é IDÊNTICO nos quatro arquivos de
# família (baseline, tree_family, gp_family, neural_family). É cópia
# deliberada, não descuido: a decisão de projeto é que os arquivos sejam
# independentes entre si (um não importa o outro), para que cada família
# possa ser lida e usada isolada. O preço é esta repetição de ~15 linhas.
# tests/metamodels/test_sample_weight.py trava as quatro cópias como
# byte-idênticas, então elas não podem divergir em silêncio.

def fit_with_sample_weight(estimator, X, y, sample_weight=None):
    """Ajusta `estimator`, repassando sample_weight só a quem aceita.

    De onde vem o peso: a coluna f"{kpi}_sample_weight" que
    replication.aggregate_replications() calcula como 1/variância das
    replicações daquele ponto. Ponto mais ruidoso pesa menos no ajuste.

    Quem não aceita peso (ex.: GaussianProcessRegressor do sklearn) é
    ajustado SEM ele, com aviso — nunca em silêncio, e nunca com uma
    estratégia de ponderação inventada aqui para compensar (reamostragem,
    replicação de linhas etc.). Se o estimador não sabe usar peso, o
    resultado tem de deixar isso visível na comparação da Fase 4.
    """
    if sample_weight is None:
        return estimator.fit(X, y)

    destino = _destino_do_peso(estimator)
    if destino is None:
        warnings.warn(
            f"{type(estimator).__name__} não aceita sample_weight em fit(): "
            "o modelo será ajustado SEM ponderação. A comparação da Fase 4 "
            "precisa levar isso em conta.",
            UserWarning,
            stacklevel=2,
        )
        return estimator.fit(X, y)

    return estimator.fit(X, y, **{destino: sample_weight})


def _destino_do_peso(estimator) -> str | None:
    """Nome do argumento por onde o peso entra, ou None se não houver.

    Pipeline é o caso que justifica a função: Pipeline.fit() não tem um
    parâmetro `sample_weight` (has_fit_parameter devolve False), mas
    encaminha `<passo>__sample_weight` para o passo final. Sem este
    roteamento, os baselines polinomiais — que SÃO pipelines — perderiam o
    peso caladinhos.
    """
    if isinstance(estimator, Pipeline):
        nome, final = estimator.steps[-1]
        return f"{nome}__sample_weight" if has_fit_parameter(final, "sample_weight") else None
    return "sample_weight" if has_fit_parameter(estimator, "sample_weight") else None


# ---------------------------------------------------------------------
# 1. Dummy — o piso absoluto
# ---------------------------------------------------------------------

def build_dummy_mean() -> DummyRegressor:
    """Prevê sempre a média de y_train, ignorando X inteiramente.

    É o piso absoluto do projeto: por construção seu R² é 0 no conjunto em
    que foi ajustado (a soma dos quadrados dos resíduos iguala a soma total).
    Num conjunto de teste o R² costuma ficar LIGEIRAMENTE NEGATIVO, porque a
    média do treino não é a média do teste — isso é esperado e não é bug.

    Serve de teste de sanidade: metamodelo que não bate o dummy está pior do
    que chutar a média, e nesse caso o problema não é escolha de família — é
    dado, DoE ou ruído. Aceita sample_weight (a média vira média ponderada).
    """
    return DummyRegressor(strategy="mean")


#: Não há o que tunar: strategy="mean" não tem hiperparâmetro. O grid vazio
#: existe para o dummy passar pelo mesmo GridSearchCV que os outros, sem
#: precisar de caso especial em quem chama.
GRID_DUMMY_MEAN: dict[str, list] = {}


# ---------------------------------------------------------------------
# 2. Linear
# ---------------------------------------------------------------------

def build_linear() -> LinearRegression:
    """Regressão linear pura — o efeito principal de cada parâmetro, sem
    curvatura nem interação.

    Interpretação direta no nosso pipeline: como X chega escalado para [0,1]
    pelos limites DECLARADOS no ParameterSpec, cada coeficiente é o efeito
    de varrer aquele parâmetro de `low` a `high` — comparável entre
    parâmetros de unidades diferentes sem padronização extra.
    """
    return LinearRegression()


#: Sem hiperparâmetro (fit_intercept=True é o certo aqui: y padronizado no
#: treino tem média zero, mas o TESTE não, e forçar intercepto zero jogaria
#: fora esse grau de liberdade sem ganho nenhum).
GRID_LINEAR: dict[str, list] = {}


# ---------------------------------------------------------------------
# 3. Polinomial
# ---------------------------------------------------------------------

def build_polynomial(degree: int = 2) -> Pipeline:
    """Regressão polinomial: captura curvatura e interações de 2ª ordem.

    É a superfície de resposta clássica de DoE (RSM). Se ela já resolve o
    problema, ver o item 2 do docstring do módulo: é RESULTADO, não fracasso.

    include_bias=False porque o LinearRegression já ajusta o intercepto —
    com bias nas duas pontas a coluna constante fica colinear com o
    intercepto e a solução vira indeterminada (o lstsq resolve, mas os
    coeficientes deixam de ser interpretáveis um a um).

    O passo final chama-se "reg" — é esse nome que fit_with_sample_weight()
    usa para rotear o peso, e é o prefixo das chaves do grid.
    """
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", LinearRegression()),
    ])


#: degree=1 entra de propósito: deixa o GridSearchCV decidir se a curvatura
#: paga o custo, em vez de assumir que paga.
GRID_POLYNOMIAL: dict[str, list] = {
    "poly__degree": [1, 2, 3],
}


# ---------------------------------------------------------------------
# 4. Ridge — e o polinomial regularizado
# ---------------------------------------------------------------------

def build_ridge(alpha: float = 1.0) -> Ridge:
    """Regressão linear com penalização L2.

    No regime linear puro a regularização raramente muda muito (há poucos
    termos), mas ela é a referência que torna interpretável o ganho do
    polinomial regularizado abaixo.
    """
    return Ridge(alpha=alpha)


#: Faixa larga em escala log: com y padronizado pelo preprocessing, alpha da
#: ordem de 1 já é uma penalização apreciável, e os extremos cobrem tanto o
#: "quase sem regularização" quanto o "quase só intercepto".
GRID_RIDGE: dict[str, list] = {
    "alpha": np.logspace(-4, 3, 15).tolist(),
}


def build_polynomial_ridge(degree: int = 2, alpha: float = 1.0) -> Pipeline:
    """Polinomial COM regularização — a variante que de fato se sustenta no
    nosso regime de dados.

    POR QUE A REGULARIZAÇÃO É NECESSÁRIA AQUI: PolynomialFeatures(2) gera
    p + p(p+1)/2 termos para p parâmetros de entrada. Note que p é o número
    de COLUNAS DEPOIS do one-hot, não o número de ParameterSpec — uma
    categórica de 3 níveis já entra como 3 colunas. Alguns números do nosso
    caso:

        p =  6 colunas ->  27 termos
        p = 10 colunas ->  65 termos
        p = 15 colunas -> 135 termos
        p = 20 colunas -> 230 termos

    Com os ~200 pontos de DoE que o projeto planeja, p=20 já deixa o número
    de termos da mesma ordem do número de observações: a matriz fica mal
    condicionada, os coeficientes disparam e o ajuste passa a modelar o
    ruído do simulador. O Ridge é o que mantém isso estável — e, junto com
    o teto de ruído da Fase 4, é o que permite dizer se o polinomial parou
    por limitação dele ou por limite físico do dado.

    ESCALA DOS TERMOS: a penalização L2 é sensível a escala, e aqui não há
    StandardScaler no meio de propósito — X já chega de
    preprocessing.transform_X() escalado para [0,1] pelos limites do
    ParameterSpec, então todo termo polinomial também fica em [0,1] e nenhum
    domina a penalização por unidade. Se algum dia X deixar de vir
    normalizado (ex.: alguém treinar direto no dataset bruto), insira um
    StandardScaler entre "poly" e "reg" — senão o alpha ótimo passa a
    depender das unidades de cada parâmetro.
    """
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", Ridge(alpha=alpha)),
    ])


#: Grade cruzada: o grau certo depende de quanta regularização há, e
#: vice-versa — buscar os dois eixos juntos é o ponto da variante.
GRID_POLYNOMIAL_RIDGE: dict[str, list] = {
    "poly__degree": [1, 2, 3],
    "reg__alpha": np.logspace(-4, 3, 15).tolist(),
}
