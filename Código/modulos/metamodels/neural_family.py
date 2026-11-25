"""
Família NEURAL: perceptron multicamadas (MLP) do scikit-learn.

Foi o metamodelo de melhor desempenho nos três cenários da monografia e é o
usado em framework.ipynb. Treina e prevê no espaço TRANSFORMADO pelo
pré-processamento: entradas em [0, 1] pelos limites declarados no YAML e
indicador padronizado. A escala [0, 1] das entradas é o que dispensa um
scaler adicional antes da rede.
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

def fit_with_sample_weight(estimator, X, y, sample_weight=None):
    """Ajusta `estimator`, repassando sample_weight só a quem aceita.

    O peso é opcional (no framework.ipynb o treino é feito sobre as
    réplicas, sem peso). Com peso, ponto mais ruidoso pesa menos no ajuste.

    Quem não aceita peso (ex.: GaussianProcessRegressor do sklearn) é
    ajustado SEM ele, com aviso — nunca em silêncio, e nunca com uma
    estratégia de ponderação inventada aqui para compensar (reamostragem,
    replicação de linhas etc.). Se o estimador não sabe usar peso, o
    resultado tem de deixar isso visível na comparação entre modelos.
    """
    if sample_weight is None:
        return estimator.fit(X, y)

    destino = _destino_do_peso(estimator)
    if destino is None:
        warnings.warn(
            f"{type(estimator).__name__} não aceita sample_weight em fit(): "
            "o modelo será ajustado SEM ponderação. A comparação entre modelos "
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
    alpha: float = 1e-1,
    activation: str = "relu",
    solver: str = "lbfgs",
    max_iter: int = 5000,
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

    alpha=1e-1 em vez do default 1e-4 do sklearn: penalização L2 três
    ordens de grandeza maior. Com resposta ruidosa e poucos pontos, a rede
    tem capacidade de sobra para memorizar o ruído de replicação, e a
    regularização é o único freio. É o hiperparâmetro mais importante do
    grid, e o default foi calibrado por medição. Num sintético de 200
    pontos com piso de ruído 0.1445 (em unidades de desvio de y), varrendo
    alpha com o otimizador rodando até convergir de verdade:

        alpha    RMSE treino   RMSE teste
        1e-4        0.097        0.181
        1e-2        0.067        0.181
        1e-1        0.141        0.147
        3e-1        0.149        0.138   <- bate o piso de ruído
        1.0         0.156        0.150

    O padrão é inequívoco: abaixo de 1e-1 a rede desce no treino e SOBE no
    teste, que é a assinatura de estar ajustando ruído.

    max_iter=5000 porque lbfgs conta ITERAÇÕES, não épocas. O número é
    generoso de propósito, e aqui vale um cuidado que custou uma medição
    para aparecer: um max_iter apertado FUNCIONA COMO REGULARIZADOR
    IMPLÍCITO (early stopping disfarçado). Com o alpha fraco de antes, o
    modelo dava RMSE de teste 0.149 parando em 200 iterações e 0.181 ao
    convergir em 2778 — ou seja, o limite de iteração estava escondendo o
    sobreajuste e fazendo o modelo parecer melhor do que era. Isso é
    especialmente ruim num GridSearchCV: a busca varre alpha achando que
    controla a regularização, enquanto um segundo regularizador não
    declarado age por baixo. Com alpha bem escolhido o problema some
    sozinho — a penalização mais forte condiciona melhor o objetivo e a
    convergência chega em ~300-600 iterações, MUITO antes do teto.

    (64, 64): duas camadas de largura moderada. Mais fundo do que isso, no
    nosso regime, adiciona parâmetros sem dado para ajustá-los. O grid
    inclui arquiteturas menores de propósito, inclusive uma camada só.

    Aceita sample_weight (sklearn >= 1.7).
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
    # 3e-1 entra para a grade CERCAR o ótimo medido (~0.3) em vez de deixá-lo
    # num vão entre 1e-1 e 1.0 — ver a tabela no docstring de build_mlp.
    # 3.0 e 10.0: no cenário 1 o alpha escolhido bateu no teto antigo (1.0)
    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
    "activation": ["relu", "tanh"],
}
