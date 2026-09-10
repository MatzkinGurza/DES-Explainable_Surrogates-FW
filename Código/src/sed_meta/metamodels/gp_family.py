"""
Fase 3 — família de PROCESSOS GAUSSIANOS: as que dão incerteza de graça.

O que torna esta família coesa (e por isso um arquivo só): todas fornecem
VARIÂNCIA PREDITIVA NATIVA. Não é detalhe de implementação — num trabalho
sobre metamodelagem de simulação ESTOCÁSTICA, saber onde o metamodelo não
sabe é metade da resposta. As outras famílias precisam de conformal
prediction (mapie) para dizer a mesma coisa; estas já dizem.

As três variantes formam uma escada deliberada sobre o tratamento do RUÍDO:

    GaussianProcessRegressor  ruído homocedástico, via WhiteKernel
    smt.KRG                   ruído homocedástico, via eval_noise
    hetGPy.hetGP              ruído HETEROCEDÁSTICO, modelado como processo

A última é a tecnicamente adequada para SED: a variância de um KPI de fila
não é constante ao longo do espaço de parâmetros — perto da saturação ela
explode. Um GP homocedástico estima UM nível de ruído médio e, com isso,
subestima a incerteza na região saturada (justamente a interessante) e a
superestima na região tranquila. As três estão aqui para que a monografia
possa MEDIR essa diferença em vez de afirmá-la.

POSIÇÃO NO PIPELINE

    preprocessing.PreprocessingPipeline -> X_train/X_test, y_train/y_test
        -> ESTE MÓDULO (treina)         -> modelo ajustado
            -> evaluation (Fase 4)      -> R², RMSE, teto de ruído
                -> xai (Fase 5)         -> SHAP/PDP

ESPAÇO TRANSFORMADO: treina e prevê no espaço JÁ TRANSFORMADO. Não se chama
inverse_transform_y aqui — Fase 4. Isso vale em dobro para o desvio-padrão
devolvido por predict(return_std=True): ele está na escala de y
TRANSFORMADO, e se o KPI usou method="log_standard" a volta para minutos
NÃO é aplicar expm1 no intervalo — é assunto da Fase 4, com o objeto de
preprocessing em mãos.

UM MODELO POR KPI: X é compartilhado, y não. O laço por KPI é de quem chama.

FORA DE ESCOPO: métricas (Fase 4), SHAP/PDP (Fase 5), laço por KPI.


=====================================================================
DUAS TENSÕES REGISTRADAS — NENHUMA RESOLVIDA NESTA FASE
=====================================================================

(a) A ESCOLHA ENTRE hetGP E smt.KRG TEM CONSEQUÊNCIA NA FASE 5.

    O pacote smt-explainability (declarado no requirements.txt do projeto)
    foi feito para explicar surrogates do PRÓPRIO SMT — ele conversa com a
    API de smt.surrogate_models. Usar KRG mantém o metamodelo dentro
    daquele ecossistema, com SHAP/PDP/Sobol integrados e testados contra o
    surrogate. Usar hetGP obriga a cair no shap genérico (KernelExplainer),
    que é agnóstico mas aproximado e caro — e que, para um GP, exige
    escolher dado de fundo e número de amostras, decisões que afetam a
    explicação e teriam de ser justificadas na monografia.

    Ou seja: a escolha do GP não é só estatística, é também sobre qual
    caminho de explicação fica disponível. Registrado aqui para NÃO ser
    descoberto por acidente na Fase 5. Decisão adiada de propósito — ela
    depende de qual das duas de fato ganhar na Fase 4.

(b) O hetGP FOI PROJETADO PARA RECEBER A ESTRUTURA DE REPLICAÇÃO, E NÓS
    ENTREGAMOS DADO JÁ AGREGADO.

    O método (Binois, Gramacy & Ludkovski) tira proveito de haver VÁRIAS
    observações no mesmo ponto do design: é comparando réplicas de um mesmo
    x que ele estima a variância local, e é isso que permite modelar o
    ruído como um processo latente sobre o espaço. O hetgpy tem até uma
    função dedicada a detectar essa estrutura (hetgpy.find_reps).

    Nosso preprocessing entrega o dataset JÁ AGREGADO por
    aggregate_replications(): uma linha por ponto, com a média e um
    sample_weight = 1/variância. Isso preserva parte da informação (a
    variância por ponto está lá) mas descarta as réplicas individuais — ou
    seja, joga fora justamente parte do que torna o hetGP a escolha certa,
    e ainda por cima entrega essa informação num formato (peso amostral)
    que o hetGP não consome.

    Consequência prática: rodado sobre dado agregado, o hetGP aqui vira
    um GP sobre médias com nugget estimado, e o ganho sobre o homocedástico
    fica menor do que o método permitiria. As saídas possíveis, para quando
    isto for enfrentado:
      - alimentar o hetGP com o `raw` por replicação (pré-agregação), o que
        exige um caminho paralelo ao preprocessing atual e um split
        AGRUPADO por point_id (a nota já registrada em
        preprocessing.pipeline.split);
      - ou passar as variâncias por ponto pelo argumento `known` do
        mleHetGP, fixando o processo de ruído em vez de estimá-lo.
    NÃO resolver nesta tarefa. O registro é para a decisão ser tomada com
    conhecimento de causa, e para que um ganho pequeno do hetGP na Fase 4
    não seja lido como "heterocedasticidade não importa" quando pode ser
    "não demos ao método o que ele pede".
"""

import warnings

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted, has_fit_parameter

#: Semente única do módulo, para comparabilidade entre variantes.
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
# 1. GaussianProcessRegressor (sklearn) — baseline da família
# ---------------------------------------------------------------------

def build_sklearn_gp(
    nu: float = 2.5,
    length_scale: float = 1.0,
    noise_level: float = 1e-2,
    n_restarts_optimizer: int = 5,
) -> GaussianProcessRegressor:
    """GP do sklearn com kernel Matérn + ruído branco.

    O WhiteKernel NÃO É OPCIONAL AQUI. Sem ele o GP interpola: passa
    exatamente por cada observação e devolve variância ~0 nos pontos de
    treino. Para resposta de simulação estocástica isso é simplesmente
    errado — cada y é uma média de replicações, com erro padrão conhecido
    e não nulo. Com o WhiteKernel, o nível de ruído vira hiperparâmetro
    estimado por máxima verossimilhança, e o GP passa a SUAVIZAR em vez de
    interpolar. (É o mesmo papel do eval_noise no KRG abaixo.)

    Matérn nu=2.5 em vez do RBF: o RBF assume superfície infinitamente
    diferenciável, o que é otimista demais para resposta de fila. nu=2.5
    (duas vezes diferenciável) é o meio-termo usual em surrogates de
    simulação; nu=1.5 é a opção mais rugosa e está no grid.

    normalize_y=False de propósito: y JÁ chega padronizado do
    preprocessing, no split de treino e só nele. Deixar o sklearn
    renormalizar aqui recentraria y usando o dado que ele recebeu — o que
    dentro de um GridSearchCV é o fold de treino, então não vaza; mas
    duplicar a padronização torna a escala do noise_level estimado difícil
    de comparar com o teto de ruído da Fase 4. Uma padronização só, a do
    preprocessing.

    NÃO aceita sample_weight (limitação do sklearn) — fit_with_sample_weight
    avisa e segue sem peso. É uma das razões de o hetGP existir nesta lista.
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3), nu=nu)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1e1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=RANDOM_STATE,
    )


#: O kernel é um objeto, então a busca se faz sobre ele inteiro — é assim
#: que o sklearn expõe isso. Varia a rugosidade (nu) e o chute inicial do
#: nível de ruído, que é o parâmetro mais sensível a mínimo local aqui.
GRID_SKLEARN_GP: dict[str, list] = {
    "kernel": [
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=nu)
        + WhiteKernel(noise_level=nl, noise_level_bounds=(1e-8, 1e1))
        for nu in (1.5, 2.5)
        for nl in (1e-3, 1e-2, 1e-1)
    ],
    "n_restarts_optimizer": [5, 10],
}


# ---------------------------------------------------------------------
# 2. hetGPy — GP heterocedástico
# ---------------------------------------------------------------------

class HetGPRegressor(BaseEstimator, RegressorMixin):
    """Wrapper de COMPATIBILIDADE do hetgpy com a API do scikit-learn.

    Existe só para que GridSearchCV e cross_val_score funcionem sem
    adaptação: o hetgpy expõe `mleHetGP(X, Z)` / `predict(x) -> dict`, e o
    sklearn espera `fit(X, y)` / `predict(X) -> array`. NÃO HÁ LÓGICA DE
    MODELO AQUI — nenhuma estatística é calculada nesta classe; tudo é
    delegado ao hetgpy. Se algo parecer decisão de modelagem neste código,
    é bug.

    Parâmetros
    ----------
    model : "hetGP" (heterocedástico) ou "homGP" (homocedástico, mesmo
        pacote). Ter os dois sob o mesmo wrapper permite isolar o ganho da
        heterocedasticidade mantendo TODO o resto igual — mesma
        implementação, mesmo otimizador, mesmo kernel. É a comparação
        limpa que a monografia precisa para afirmar que o ganho vem do
        tratamento do ruído, e não de diferença entre bibliotecas.
    covtype : kernel do hetgpy ("Gaussian", "Matern5_2", "Matern3_2").
    maxit : iterações do otimizador de verossimilhança.

    Atributos ajustados
    -------------------
    model_ : o objeto hetgpy ajustado, para quem precisar do que o wrapper
        não expõe (nugs, sd2 do processo de ruído, summary()).
    n_features_in_ : exigido pela convenção do sklearn.

    Sobre sample_weight: o hetGP não recebe peso amostral — ele estima o
    ruído por conta própria, que é o ponto do método. fit() aceita o
    argumento para manter a assinatura uniforme, mas AVISA e ignora. Ver a
    tensão (b) no docstring do módulo: o formato que ele de fato quereria
    é a estrutura de replicação, não o peso.
    """

    def __init__(
        self,
        model: str = "hetGP",
        covtype: str = "Gaussian",
        maxit: int = 100,
    ):
        # Convenção do sklearn: __init__ só ARMAZENA, sem validar nem
        # transformar. get_params/set_params (e portanto clone(), e
        # portanto GridSearchCV) dependem disso.
        self.model = model
        self.covtype = covtype
        self.maxit = maxit

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            warnings.warn(
                "hetGP não usa sample_weight: ele estima a estrutura de ruído "
                "por conta própria, que é justamente o ponto do método. O peso "
                "será ignorado. Ver a tensão (b) no docstring de gp_family.",
                UserWarning,
                stacklevel=2,
            )

        X = check_array(X, dtype="numeric", ensure_2d=True)
        y = check_array(y, dtype="numeric", ensure_2d=False).ravel()
        if len(X) != len(y):
            raise ValueError(f"X tem {len(X)} linhas e y tem {len(y)}.")

        if self.model == "hetGP":
            from hetgpy.hetGP import hetGP

            modelo = hetGP()
            ajustar = modelo.mleHetGP
        elif self.model == "homGP":
            from hetgpy.homGP import homGP

            modelo = homGP()
            ajustar = modelo.mleHomGP
        else:
            raise ValueError(
                f"model={self.model!r} desconhecido; use 'hetGP' ou 'homGP'."
            )

        ajustar(X=X, Z=y, covtype=self.covtype, maxit=self.maxit)

        self.model_ = modelo
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std: bool = False):
        """Média preditiva; com return_std=True, também o desvio-padrão.

        O desvio devolvido é o da PREVISÃO COMPLETA: o hetgpy separa `sd2`
        (variância do processo latente, sem o nugget) de `nugs` (variância
        do ruído observacional naquele x), e a documentação dele é
        explícita que a variância preditiva total é a soma das duas. É a
        soma que responde "quanto erro esperar de uma nova observação
        naqui", que é a pergunta da Fase 4 — usar só `sd2` produziria
        intervalos otimistas de forma sistemática.

        O desvio está na escala de y TRANSFORMADO — ver docstring do módulo.
        """
        check_is_fitted(self, "model_")
        X = check_array(X, dtype="numeric", ensure_2d=True)

        saida = self.model_.predict(x=X)
        media = np.asarray(saida["mean"], dtype="float64").ravel()
        if not return_std:
            return media

        sd2 = np.asarray(saida["sd2"], dtype="float64").ravel()
        nugs = np.asarray(saida["nugs"], dtype="float64").ravel()
        # clip em 0: variância numericamente negativa por arredondamento
        # perto de um ponto de treino é conhecida em GP e viraria NaN no sqrt
        return media, np.sqrt(np.clip(sd2 + nugs, 0.0, None))


def build_hetgp(
    model: str = "hetGP",
    covtype: str = "Gaussian",
    maxit: int = 100,
) -> HetGPRegressor:
    """GP heterocedástico do hetgpy — o tecnicamente adequado para SED.

    Por que ele é a escolha certa no papel: a variância do KPI muda ao longo
    do espaço de parâmetros. Num sistema de filas perto da saturação, dobrar
    a taxa de chegada não só aumenta o lead time médio como aumenta MUITO a
    dispersão entre replicações. O hetGP modela essa variância como um
    segundo processo latente sobre o espaço, em vez de um escalar.

    Por que o ganho pode vir menor do que se espera: ver a tensão (b) no
    docstring do módulo — estamos alimentando o método com dado já agregado.
    """
    return HetGPRegressor(model=model, covtype=covtype, maxit=maxit)


#: "hetGP" vs "homGP" entra no grid de propósito: é a comparação controlada
#: descrita no docstring da classe. Se o GridSearchCV escolher homGP, isso é
#: informação — mas leia junto com a tensão (b) antes de concluir.
GRID_HETGP: dict[str, list] = {
    "model": ["hetGP", "homGP"],
    "covtype": ["Gaussian", "Matern5_2", "Matern3_2"],
}


# ---------------------------------------------------------------------
# 3. smt.surrogate_models.KRG — kriging do SMT
# ---------------------------------------------------------------------

class SMTKrigingRegressor(BaseEstimator, RegressorMixin):
    """Wrapper de COMPATIBILIDADE do smt.KRG com a API do scikit-learn.

    Mesma natureza do HetGPRegressor: o SMT expõe
    `set_training_values` / `train()` / `predict_values()`, o sklearn
    espera `fit` / `predict`. NÃO HÁ LÓGICA DE MODELO AQUI.

    Parâmetros
    ----------
    corr : função de correlação ("squar_exp", "matern52", "matern32",
        "abs_exp", "pow_exp").
    poly : tendência ("constant", "linear", "quadratic") — o termo de
        regressão que o kriging universal soma ao processo.
    eval_noise : estimar o nível de ruído em vez de interpolar. Default
        True aqui, ao contrário do default do SMT — ver build_smt_kriging().
    n_start : reinícios do otimizador de hiperparâmetros (mínimos locais na
        verossimilhança são o problema prático número um em kriging).
    theta0 : chute inicial da escala de correlação. None deixa o default do
        SMT (0.01), que ele transmite a todas as dimensões.

    Atributos ajustados
    -------------------
    model_ : o objeto KRG ajustado.
    n_features_in_ : exigido pela convenção do sklearn.

    Sobre sample_weight: o KRG não expõe peso amostral. fit() aceita o
    argumento para manter a assinatura uniforme, mas avisa e ignora.
    """

    def __init__(
        self,
        corr: str = "squar_exp",
        poly: str = "constant",
        eval_noise: bool = True,
        n_start: int = 10,
        theta0: float | None = None,
    ):
        # Só armazena — ver a mesma nota em HetGPRegressor.__init__.
        self.corr = corr
        self.poly = poly
        self.eval_noise = eval_noise
        self.n_start = n_start
        self.theta0 = theta0

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            warnings.warn(
                "smt.KRG não aceita sample_weight; o peso será ignorado. Para "
                "ponderar por ruído nesta família, use eval_noise=True (que "
                "estima o ruído) ou o hetGP.",
                UserWarning,
                stacklevel=2,
            )

        from smt.surrogate_models import KRG

        X = check_array(X, dtype="numeric", ensure_2d=True)
        y = check_array(y, dtype="numeric", ensure_2d=False).ravel()
        if len(X) != len(y):
            raise ValueError(f"X tem {len(X)} linhas e y tem {len(y)}.")

        opcoes = dict(
            corr=self.corr,
            poly=self.poly,
            eval_noise=self.eval_noise,
            n_start=self.n_start,
            # o SMT é falante por default; num GridSearchCV isso inunda a
            # saída com um relatório por fold e por combinação
            print_global=False,
        )
        if self.theta0 is not None:
            opcoes["theta0"] = [self.theta0] * X.shape[1]

        modelo = KRG(**opcoes)
        modelo.set_training_values(X, y)
        modelo.train()

        self.model_ = modelo
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std: bool = False):
        """Média preditiva; com return_std=True, também o desvio-padrão.

        O SMT devolve colunas (n, 1); o sklearn espera (n,) em regressão de
        alvo único — daí o ravel(). O desvio está na escala de y
        TRANSFORMADO, ver docstring do módulo.
        """
        check_is_fitted(self, "model_")
        X = check_array(X, dtype="numeric", ensure_2d=True)

        media = np.asarray(self.model_.predict_values(X), dtype="float64").ravel()
        if not return_std:
            return media

        var = np.asarray(self.model_.predict_variances(X), dtype="float64").ravel()
        # mesmo motivo do clip no HetGPRegressor
        return media, np.sqrt(np.clip(var, 0.0, None))


def build_smt_kriging(
    corr: str = "squar_exp",
    poly: str = "constant",
    eval_noise: bool = True,
    n_start: int = 10,
) -> SMTKrigingRegressor:
    """Kriging do SMT — homocedástico, e dentro do ecossistema do SMT.

    eval_noise=True É DESVIO DELIBERADO DO DEFAULT DO SMT (que é False).
    Com eval_noise=False o KRG trata a resposta como EXATA: não estima
    nível de ruído, e o único amortecimento que sobra é o `nugget`
    numérico. O sintoma prático mais confiável não é o RMSE de treino — o
    otimizador de hiperparâmetros às vezes encontra sozinho uma solução
    suave, e num sintético de 3 fatores medido ao montar este módulo os
    dois ajustes deram RMSE de treino parecido (0.153 contra 0.150) — e sim
    a INCERTEZA REPORTADA: com eval_noise=False o desvio preditivo médio
    caiu para cerca de 60% do outro (0.016 contra 0.028 nos pontos de
    treino). Ou seja, o modelo fica confiante demais.

    Isso importa porque a variância preditiva é metade do produto desta
    família (ver docstring do módulo): um KRG que subestima a própria
    incerteza atrapalha exatamente onde ele deveria ajudar. Em espaços de
    dimensão maior o efeito pode ir além disso e virar interpolação de
    fato, com RMSE de treino ~0 — observado num probe de 5 fatores.

    Mesmo motivo do WhiteKernel em build_sklearn_gp(). Só volte a False se
    algum dia este metamodelo for usado sobre resposta determinística.

    Lembrete da tensão (a) no docstring do módulo: esta é a variante que
    mantém a Fase 5 dentro do smt-explainability.
    """
    return SMTKrigingRegressor(
        corr=corr, poly=poly, eval_noise=eval_noise, n_start=n_start
    )


#: eval_noise fica FORA do grid de propósito: não é hiperparâmetro a
#: otimizar, é premissa do problema (a resposta é ruidosa). Deixá-lo no
#: grid convidaria o CV a escolher a interpolação num fold sortudo.
GRID_SMT_KRIGING: dict[str, list] = {
    "corr": ["squar_exp", "matern52", "matern32"],
    "poly": ["constant", "linear"],
    "n_start": [10, 20],
}
