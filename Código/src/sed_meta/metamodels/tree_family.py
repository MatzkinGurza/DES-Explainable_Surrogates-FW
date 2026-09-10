"""
Fase 3 — família de ÁRVORES: ensembles baseados em particionamento.

O que torna esta família coesa (e por isso um arquivo só):

  - todas aceitam sample_weight em fit() com a mesma semântica;
  - todas são compatíveis com TreeSHAP na Fase 5 — SHAP exato em tempo
    polinomial, em vez do KernelSHAP aproximado e caro que as outras
    famílias exigem. Isso é um argumento prático forte a favor desta
    família num trabalho cujo objetivo é EXPLICAR o metamodelo;
  - todas capturam interação e não-linearidade sem que ninguém precise
    declarar termos, ao contrário do polinomial do baseline;
  - e todas são invariantes a transformação monotônica de X, então a
    escala [0,1] que o preprocessing aplica não as afeta (o que não quer
    dizer que atrapalhe — ver nota abaixo).

POSIÇÃO NO PIPELINE

    preprocessing.PreprocessingPipeline -> X_train/X_test, y_train/y_test
        -> ESTE MÓDULO (treina)         -> modelo ajustado
            -> evaluation (Fase 4)      -> R², RMSE, teto de ruído
                -> xai (Fase 5)         -> TreeSHAP / PDP

ESPAÇO TRANSFORMADO: treina e prevê no espaço JÁ TRANSFORMADO pelo
preprocessing. Não se chama inverse_transform_y aqui — reportar em
minutos/peças é da Fase 4.

UM MODELO POR KPI: X é compartilhado, y não. O laço por KPI é de quem chama.

NOTA SOBRE O ONE-HOT: árvores lidariam nativamente com categóricas, e o
one-hot que o preprocessing aplica fragmenta um fator de k níveis em k
testes binários — o que dilui a importância daquele fator entre as k
colunas na Fase 5. É um custo conhecido e aceito: o preço de ter UM X só,
determinístico e compartilhado por todas as famílias (várias das quais
exigem numérico). Na leitura do SHAP, some as contribuições das colunas do
mesmo fator antes de comparar com um parâmetro numérico.

FORA DE ESCOPO: métricas (Fase 4), SHAP/PDP (Fase 5), laço por KPI.
"""

import warnings

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter
from xgboost import XGBRegressor

#: Semente única do módulo, para que as quatro variantes sejam comparáveis
#: entre si e reprodutíveis entre execuções (RNF de rastreabilidade).
RANDOM_STATE = 42


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
# O CUIDADO QUE ORGANIZA OS GRIDS DESTA FAMÍLIA
# ---------------------------------------------------------------------
# BAGGING (RandomForest, ExtraTrees) NÃO SE TUNA EM n_estimators.
#
# Breiman (2001) mostra que o erro de generalização do Random Forest
# converge conforme o número de árvores cresce — ele NÃO sobreajusta com
# mais árvores, só para de melhorar. Logo, incluir n_estimators no
# GridSearchCV é gastar orçamento computacional num eixo cuja resposta já
# se conhece: "mais é melhor, com retorno decrescente". A prática correta é
# FIXAR n_estimators num valor generoso (o custo é linear e o ganho satura)
# e buscar sobre o que de fato controla o viés-variância de cada árvore:
#
#       max_depth, min_samples_leaf, max_features
#
# BOOSTING É O CONTRÁRIO — e a distinção importa. Em GradientBoosting e
# XGBoost, cada árvore corrige o resíduo da anterior, então n_estimators é
# o próprio número de passos de descida: passar do ponto SOBREAJUSTA de
# verdade. Ali n_estimators é eixo legítimo de busca, acoplado ao
# learning_rate (mais árvores pedem passo menor). Por isso os grids abaixo
# tratam as duas subfamílias de forma deliberadamente diferente — não é
# inconsistência de quem escreveu.
#
# REGIME DE DADOS (poucas centenas de pontos, ruído alto do simulador): os
# defaults do sklearn levam a árvore até a folha pura (min_samples_leaf=1),
# o que aqui significa memorizar o ruído de replicação. Os defaults abaixo
# partem de folhas maiores de propósito.


# ---------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------

def build_random_forest(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 3,
    max_features: float | str = 1.0,
) -> RandomForestRegressor:
    """Bagging de árvores com amostragem de features por split.

    n_estimators=500 é FIXO por decisão de projeto (ver bloco acima): 500 é
    generoso para algumas centenas de pontos, e não está no grid.

    min_samples_leaf=3 em vez do default 1: com ruído de replicação alto,
    folha unitária memoriza a réplica em vez de estimar a média do ponto.

    max_features=1.0 (todas as colunas em cada split) em vez do default do
    sklearn para regressão: com poucas dezenas de colunas — que é o nosso
    caso depois do one-hot — restringir demais só aumenta o viés. Continua
    no grid, porque é o principal eixo de descorrelação entre as árvores.

    n_jobs=-1: o custo aqui é paralelizável e o DoE já foi a parte cara.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


#: Sem n_estimators — ver o bloco "O CUIDADO QUE ORGANIZA OS GRIDS".
GRID_RANDOM_FOREST: dict[str, list] = {
    "max_depth": [None, 4, 8, 16],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": [1.0, 0.5, "sqrt"],
}


# ---------------------------------------------------------------------
# 2. Extra Trees
# ---------------------------------------------------------------------

def build_extra_trees(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 3,
    max_features: float | str = 1.0,
) -> ExtraTreesRegressor:
    """Como o Random Forest, mas com o ponto de corte de cada split sorteado
    em vez de otimizado.

    Por que ela ganha lugar ao lado do RF num trabalho sobre SED: o corte
    aleatório aumenta o viés e reduz a variância, e é justamente a variância
    que explode quando se ajusta árvore sobre resposta ruidosa. Em regime de
    ruído alto — o nosso, perto da saturação de fila — ExtraTrees costuma
    empatar ou ganhar do RF, e treina mais rápido. Vale medir, não assumir.

    Mesma regra de n_estimators do RF: fixo, fora do grid.
    """
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


#: Sem n_estimators, pelo mesmo motivo do RF (também é bagging).
GRID_EXTRA_TREES: dict[str, list] = {
    "max_depth": [None, 4, 8, 16],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_features": [1.0, 0.5, "sqrt"],
}


# ---------------------------------------------------------------------
# 3. Gradient Boosting (sklearn)
# ---------------------------------------------------------------------

def build_gradient_boosting(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    subsample: float = 0.8,
) -> GradientBoostingRegressor:
    """Boosting sequencial: cada árvore ajusta o resíduo da anterior.

    AQUI n_estimators ENTRA NO GRID, ao contrário do RF/ExtraTrees — é o
    número de passos de correção, e passar do ponto sobreajusta de verdade.
    Ele é acoplado ao learning_rate: o produto dos dois é aproximadamente o
    "quanto de correção total" o modelo aplica, então buscar os dois juntos
    é o certo.

    max_depth=3: em boosting cada árvore é um aprendiz FRACO de propósito —
    a profundidade limita a ordem de interação por passo (3 ≈ interações de
    até 3 fatores), e a força vem do número de passos, não do tamanho da
    árvore.

    subsample=0.8 liga o gradient boosting estocástico: além de regularizar,
    reduz a chance de o modelo perseguir o ruído de replicação de um ponto
    específico.
    """
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=RANDOM_STATE,
    )


#: n_estimators PRESENTE de propósito (boosting sobreajusta com passos
#: demais), acoplado ao learning_rate.
GRID_GRADIENT_BOOSTING: dict[str, list] = {
    "n_estimators": [100, 300, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "min_samples_leaf": [1, 3, 5],
    "subsample": [0.8, 1.0],
}


# ---------------------------------------------------------------------
# 4. XGBoost
# ---------------------------------------------------------------------

def build_xgboost(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    min_child_weight: float = 3.0,
) -> XGBRegressor:
    """Boosting com regularização explícita no objetivo.

    O que ele acrescenta ao GradientBoostingRegressor, e por isso justifica
    a quarta variante: o termo de penalização (reg_lambda, reg_alpha) entra
    na própria função de custo, em vez de vir só do formato da árvore. Em
    dado ruidoso e escasso — o nosso — isso costuma ser a diferença entre
    boosting utilizável e boosting que persegue ruído.

    min_child_weight=3 é o análogo do min_samples_leaf aqui (com peso
    unitário, é o número mínimo de observações por folha). Sobe em relação
    ao default 1 pelo mesmo motivo do RF: não memorizar réplica.

    tree_method="hist" é o algoritmo rápido e é o default moderno; fica
    explícito para o resultado não mudar se o default do pacote mudar —
    reprodutibilidade é RNF do projeto.
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


#: Também com n_estimators (boosting), mais os eixos de regularização que
#: são a razão de ser desta variante.
GRID_XGBOOST: dict[str, list] = {
    "n_estimators": [100, 300, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "min_child_weight": [1.0, 3.0, 5.0],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
}
