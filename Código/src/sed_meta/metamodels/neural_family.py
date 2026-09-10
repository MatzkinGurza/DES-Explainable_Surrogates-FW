"""
Fase 3 — família NEURAL: MLP e só.

DECISÃO DE ESCOPO REGISTRADA: NÃO ADICIONE PYTORCH AQUI.

O fichamento de referências do projeto é explícito em não sobreinvestir
tempo nesta família, e a razão é concreta: redes neurais rendem quando há
muitos dados, e o nosso regime é de POUCAS CENTENAS de pontos de DoE —
cada um custando uma rodada de ProModel com replicações. Nesse regime o
MLP raramente ganha das árvores ou do GP, e uma implementação em PyTorch
traria laço de treino, escolha de otimizador, early stopping, device e
mais um eixo de reprodutibilidade para gerenciar, tudo isso para uma
família que provavelmente vai figurar na tabela comparativa como "testada
e não competitiva" — o que já é um resultado válido, e o MLPRegressor do
sklearn o entrega com uma linha.

O gatilho para reconsiderar é UM só e ainda não foi observado: o MLP do
sklearn se mostrar competitivo E limitado pela própria implementação
(ex.: precisar de arquitetura que hidden_layer_sizes não expressa, ou de
função de perda customizada para ponderar por variância). Enquanto isso
não acontecer na prática, PyTorch é escopo que não se abre.

POSIÇÃO NO PIPELINE

    preprocessing.PreprocessingPipeline -> X_train/X_test, y_train/y_test
        -> ESTE MÓDULO (treina)         -> modelo ajustado
            -> evaluation (Fase 4)      -> R², RMSE, teto de ruído
                -> xai (Fase 5)         -> SHAP (KernelExplainer) / PDP

ESPAÇO TRANSFORMADO: treina e prevê no espaço JÁ TRANSFORMADO. Não se chama
inverse_transform_y aqui — Fase 4.

POR QUE O PREPROCESSING IMPORTA MAIS PARA ESTA FAMÍLIA QUE PARA AS OUTRAS:
árvores são invariantes a transformação monotônica de X, GPs se ajustam via
escalas de correlação. O MLP não: com entradas em escalas díspares, os
gradientes ficam desbalanceados e o treino trava em mínimo ruim. A escala
[0,1] que preprocessing.transform_X() aplica — pelos limites DECLARADOS no
ParameterSpec, não pelos observados — é o que torna esta família viável
sem um scaler extra. Não empilhe outro StandardScaler na frente.

UM MODELO POR KPI: X é compartilhado, y não. O laço por KPI é de quem chama.

FORA DE ESCOPO: métricas (Fase 4), SHAP/PDP (Fase 5), laço por KPI.
"""

import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter

#: Semente única do módulo, para comparabilidade entre variantes. Vale
#: menos aqui que nas outras famílias: a inicialização dos pesos torna o
#: MLP a família mais sensível a semente do projeto — se a Fase 4 mostrar
#: dispersão grande entre sementes, isso é resultado a reportar, não ruído
#: a esconder.
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
# MLP
# ---------------------------------------------------------------------

def build_mlp(
    hidden_layer_sizes: tuple[int, ...] = (64, 64),
    alpha: float = 1e-2,
    activation: str = "relu",
    solver: str = "lbfgs",
    max_iter: int = 2000,
) -> MLPRegressor:
    """Perceptron multicamadas do sklearn, com defaults para POUCO DADO.

    solver="lbfgs" em vez do default "adam", e isto é a escolha que mais
    importa nesta função. O adam é um otimizador estocástico pensado para
    milhares de amostras em minilotes; com algumas centenas de pontos ele
    converge devagar e de forma ruidosa. O lbfgs é de segunda ordem, usa o
    conjunto inteiro de uma vez e é o que a própria documentação do sklearn
    recomenda para dataset pequeno. Medido durante a construção deste
    módulo, num sintético de 120 pontos: RMSE 0.069 com lbfgs contra 0.119
    com adam, mesma arquitetura e mesmo orçamento.
    Consequência a lembrar no grid: com lbfgs os parâmetros de minilote
    (learning_rate_init, batch_size, early_stopping) são IGNORADOS — pô-los
    na busca só multiplica o custo sem mudar resultado nenhum.

    alpha=1e-2 em vez do default 1e-4: penalização L2 duas ordens de
    grandeza maior. Com resposta ruidosa e poucos pontos, a rede tem
    capacidade de sobra para memorizar o ruído de replicação; a
    regularização é o que segura isso. É o hiperparâmetro mais importante
    do grid.

    (64, 64): duas camadas de largura moderada. Mais fundo do que isso, no
    nosso regime, adiciona parâmetros sem dado para ajustá-los. O grid
    inclui arquiteturas menores de propósito, inclusive uma camada só.

    max_iter=2000 porque lbfgs conta ITERAÇÕES, não épocas; o default 200
    costuma parar antes de convergir e produzir um ConvergenceWarning que
    seria fácil confundir com "a família não serve".

    Aceita sample_weight (sklearn >= 1.7), então o peso por ponto do
    aggregate_replications() chega ao treino como nas árvores.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )


#: Sem learning_rate_init/batch_size/early_stopping: com solver="lbfgs" eles
#: são ignorados (ver docstring). alpha varre uma faixa larga em escala log
#: por ser o eixo que de fato controla o sobreajuste aqui.
GRID_MLP: dict[str, list] = {
    "hidden_layer_sizes": [(32,), (64,), (32, 32), (64, 64), (128, 64)],
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "activation": ["relu", "tanh"],
}
