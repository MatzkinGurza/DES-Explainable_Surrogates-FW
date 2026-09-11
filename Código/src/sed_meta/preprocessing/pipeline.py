"""
Ponte Fase 1/2 -> Fase 3 — preparação do dataset para os metamodelos.

Posição no pipeline:

    doe.replication.aggregate_replications()   -> dataset agregado
                                                  (1 linha por ponto do DoE)
        -> ESTE MÓDULO                         -> X_train/X_test, y_train/y_test
            -> metamodels (Fase 3)             -> modelo treinado
                -> evaluation (Fase 4) / xai (Fase 5)

O PRINCÍPIO QUE ORGANIZA O MÓDULO
---------------------------------
Existem aqui duas famílias de transformação, e elas têm regras OPOSTAS
sobre o que podem aprender dos dados. Confundir as duas é o erro que
este módulo existe para tornar impossível.

1) ENTRADAS (X) — determinísticas, ancoradas no ParameterSpec.

   Os limites de um parâmetro de entrada são conhecidos ANTES de qualquer
   simulação rodar: `low`/`high` de um float/integer, `levels` de um
   categorical/ordinal. Vêm da definição do espaço de busca (o YAML do
   cenário), não dos dados coletados. Por isso NÃO se usa
   MinMaxScaler().fit(X_train) — que aprenderia o min/max *observado* na
   amostra —, e sim os limites *declarados*; idem para categóricas, com
   OneHotEncoder(categories=[spec.levels]) explícito, nunca inferido.

   Isso resolve três problemas de uma vez:
     a. Vazamento sutil e permanente — o range observado no treino depende
        de quais pontos do LHS caíram naquele split; é acidente amostral,
        não propriedade do problema.
     b. Quebra na inferência — um ponto novo fora do range observado no
        treino extrapolaria de forma inconsistente, ou estouraria numa
        categoria não vista.
     c. Instabilidade entre execuções — com DoE incremental, o encoder
        mudaria de definição a cada rodada, invalidando comparações.

   Propriedade garantida (e testada em tests/preprocessing/): ajustar o
   transformador de X em QUALQUER subconjunto das linhas — inclusive uma
   única linha — produz exatamente a mesma transformação. É isso que torna
   o pipeline reproduzível a partir apenas dos specs.

2) SAÍDAS (y) — aprendidas, e SÓ do split de treino.

   O range de um KPI como lead_time_mean ou wip_mean só existe depois de
   rodar simulação: não há valor declarado a priori. Aqui a transformação
   precisa mesmo ser ajustada aos dados, e por isso a ORDEM é crítica:

       1. split treino/teste nos dados BRUTOS      -> split()
       2. fit dos transformadores SÓ no treino     -> fit(train_df)
       3. transform (NUNCA fit_transform) nos dois -> transform_y(...)

   Inverter essa ordem é vazamento clássico: estatísticas do teste
   contaminam o treino e as métricas da Fase 4 ficam otimistas de forma
   inválida — o que comprometeria a comparação contra o teto de ruído de
   replication.compute_noise_ceiling(), um dos argumentos centrais do
   trabalho.

   Nota de domínio: KPIs de tempo de fila costumam ser fortemente
   enviesados à direita, especialmente perto da saturação. Daí o método
   "log_standard" (log1p antes da padronização). É escolha EXPLÍCITA por
   KPI, nunca default automático.

CONTRATO DE COLUNAS DO DATASET DE ENTRADA
-----------------------------------------
O DataFrame esperado é a saída de aggregate_replications(): uma linha por
ponto, com

    point_id | <colunas de parâmetro> | <por KPI>: _mean, _std, _n, _sem,
                                                   _sample_weight

Duas consequências práticas:

  - O alvo de um KPI é a coluna f"{kpi}_mean", NÃO a coluna `kpi` pura.
    Ver _resolve_target_column() para o detalhe (há um fallback documentado
    para datasets não agregados, usado em teste/exploração).
  - As colunas extras (point_id, _std, _n, _sem, _sample_weight) são
    ignoradas na montagem de X pelo remainder="drop" do ColumnTransformer.
    Elas NÃO são destruídas: split() devolve DataFrames completos, então
    f"{kpi}_sample_weight" continua acessível para a Fase 3 passar ao
    `fit` do metamodelo (hetGP, regressão ponderada).

    SOBRE O PESO AMOSTRAL, e este ponto custou uma medição para aparecer:
    NÃO passe a coluna f"{kpi}_sample_weight" crua ao metamodelo quando o
    KPI usa method="log_standard". O peso é 1/variância medida na escala
    BRUTA, e o modelo treina na escala transformada — misturar as duas
    desfaz o ajuste. Use transform_sample_weight(), que converte o peso
    para a escala em que o alvo é de fato treinado. O docstring daquele
    método traz a medição do estrago (R² de teste caindo de 0,75 para
    -0,71 num cenário real do projeto).

FORA DE ESCOPO deste módulo: treino de metamodelos, hiperparâmetros,
métricas e XAI. Ele termina em X_train/X_test, y_train/y_test e no objeto
serializável que garante que qualquer previsão futura passe exatamente
pela mesma transformação.
"""

from dataclasses import dataclass, field
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from sed_meta.doe.parameter_space import ParameterSpec

logger = logging.getLogger(__name__)

#: Métodos aceitos em TargetTransformSpec.method.
TARGET_METHODS = ("standard", "log_standard", "none")

#: Sufixo da coluna que carrega a média das replicações de um KPI — é ela
#: o alvo do metamodelo, não o nome "puro" do KPI (ver docstring do módulo).
TARGET_SUFFIX = "_mean"

#: Sufixo da coluna de peso amostral (1/variância) de um KPI, produzida por
#: replication.aggregate_replications(). Ver transform_sample_weight().
WEIGHT_SUFFIX = "_sample_weight"


# =====================================================================
# 1. Transformação das ENTRADAS (X) — determinística, vinda dos specs
# =====================================================================
#
# POR QUE CLASSES CHAMÁVEIS E NÃO lambdas: save() serializa o pipeline
# inteiro com joblib (pickle), e pickle NÃO consegue serializar função
# aninhada/lambda — ele guarda a referência pelo caminho do módulo, e
# `_make_fixed_minmax.<locals>.<lambda>` não é acessível de fora. Escrever
# esses transformadores com `lambda X, lo=low, sp=span: ...` funciona em
# memória e só quebra na hora de salvar, com PicklingError. Instâncias de
# classe definida no nível do módulo picklam sem problema (o pickle guarda
# a classe pelo nome e os parâmetros no __dict__), então os limites
# continuam viajando com o objeto. Não troque por lambda.


@dataclass(frozen=True)
class _ScaleToUnit:
    """(X - low) / span — leva o intervalo declarado para [0,1]."""
    low: float
    span: float

    def __call__(self, X):
        return (X - self.low) / self.span


@dataclass(frozen=True)
class _ScaleFromUnit:
    """Inversa de _ScaleToUnit: Z * span + low."""
    low: float
    span: float

    def __call__(self, Z):
        return Z * self.span + self.low


@dataclass(frozen=True)
class _DivideBy:
    """X / denom — usada para levar o código ordinal (0..n-1) a [0,1]."""
    denom: float

    def __call__(self, X):
        return X / self.denom


@dataclass(frozen=True)
class _MultiplyBy:
    """Inversa de _DivideBy: Z * denom."""
    denom: float

    def __call__(self, Z):
        return Z * self.denom


def _make_fixed_minmax(low, high) -> FunctionTransformer:
    """Escala para [0,1] usando limites DECLARADOS no spec, não observados
    nos dados.

    Não aprende nada em fit() — é exatamente isso que garante o
    determinismo e a invariância ao subconjunto de linhas: os limites
    viajam dentro do próprio transformador (`low`, `span`), então fit() é
    um no-op e transform() só depende do spec.
    """
    span = float(high) - float(low)
    if span == 0:
        raise ValueError(
            f"low e high iguais ({low}): parâmetro constante não tem escala "
            "definida. Remova-o do espaço de parâmetros."
        )
    return FunctionTransformer(
        func=_ScaleToUnit(float(low), span),
        inverse_func=_ScaleFromUnit(float(low), span),
        validate=False,
        feature_names_out="one-to-one",
    )


def _make_onehot(levels: list) -> OneHotEncoder:
    """One-hot com as categorias DECLARADAS no spec.

    handle_unknown="error" é intencional: uma categoria fora de
    spec.levels indica inconsistência entre o DoE e os dados coletados
    (planilha editada à mão, spec alterado depois de rodar). Isso deve
    estourar, não passar silenciosamente como um vetor de zeros.
    """
    return OneHotEncoder(
        categories=[list(levels)],
        handle_unknown="error",
        sparse_output=False,
    )


def _make_ordinal_scaler(levels: list) -> Pipeline:
    """Codifica um fator ORDINAL pela posição na lista declarada de níveis,
    reescalada para [0,1].

    Por que não one-hot: ordinal tem ordem natural (baixa < media < alta),
    e one-hot a jogaria fora — o metamodelo perderia a monotonicidade que
    a ordem carrega. Por que não usar os rótulos como número: eles podem
    ser texto, e mesmo quando são número (ex.: lote [10, 50, 200]) o
    espaçamento não é necessariamente uniforme; o que o spec declara é a
    ORDEM, então é a posição que se codifica.

    Continua determinístico: OrdinalEncoder recebe `categories` explícito
    (nada é inferido do dado) e o divisor vem de len(levels).
    """
    n = len(levels)
    if n < 2:
        raise ValueError(
            f"ordinal precisa de ao menos 2 níveis, recebeu {levels!r}."
        )
    denom = float(n - 1)
    return Pipeline([
        ("code", OrdinalEncoder(categories=[list(levels)], handle_unknown="error")),
        ("scale", FunctionTransformer(
            func=_DivideBy(denom),
            inverse_func=_MultiplyBy(denom),
            validate=False,
            feature_names_out="one-to-one",
        )),
    ])


def build_input_transformer(param_specs: list[ParameterSpec]) -> ColumnTransformer:
    """Monta o ColumnTransformer de X a partir dos specs — sem olhar dado.

    Um transformador por parâmetro, nomeado com o nome do parâmetro (é
    disso que sai o prefixo limpo em get_feature_names()). remainder="drop"
    descarta tudo que não for parâmetro declarado: point_id, as colunas
    agregadas por KPI e qualquer coluna extra da planilha.
    """
    if not param_specs:
        raise ValueError("param_specs vazio: não há X para construir.")

    nomes = [s.name for s in param_specs]
    duplicados = sorted({n for n in nomes if nomes.count(n) > 1})
    if duplicados:
        raise ValueError(f"param_specs com nomes repetidos: {duplicados}")

    # "__" é o separador que o ColumnTransformer usa entre o nome do
    # transformador e o da feature. O sklearn rejeita o nome, mas com uma
    # mensagem ("Estimator names must not contain __") que não diz de onde
    # veio — a origem aqui é sempre o `name` de um ParameterSpec, ou seja,
    # o YAML do cenário.
    com_sep = sorted(n for n in nomes if "__" in n)
    if com_sep:
        raise ValueError(
            f"Nome de parâmetro não pode conter '__': {com_sep}. É o separador "
            "usado pelo ColumnTransformer entre transformador e feature; "
            "renomeie o parâmetro no YAML do cenário (ex.: '_' simples)."
        )

    transformers = []
    for spec in param_specs:
        if spec.type in ("float", "integer"):
            transformer = _make_fixed_minmax(spec.low, spec.high)
        elif spec.type == "categorical":
            transformer = _make_onehot(spec.levels)
        elif spec.type == "ordinal":
            transformer = _make_ordinal_scaler(spec.levels)
        elif spec.type == "binary" and spec.ordered:
            # ordered=True vira IntegerVariable(0,1) no DoE: chega como int
            transformer = _make_fixed_minmax(0, 1)
        elif spec.type == "binary" and not spec.ordered:
            # ordered=False vira CategoricalVariable(["0","1"]) no DoE: os
            # valores chegam como STRING "0"/"1", não como int — ver
            # parameter_space._to_smt_variable e _cast_dtypes.
            transformer = _make_onehot(["0", "1"])
        else:
            raise ValueError(f"tipo de parâmetro não suportado: {spec.type!r}")

        transformers.append((spec.name, transformer, [spec.name]))

    return ColumnTransformer(transformers, remainder="drop")


# =====================================================================
# 2. Transformação das SAÍDAS (y) — aprendida, só do treino
# =====================================================================

@dataclass
class TargetTransformSpec:
    """Como transformar UM KPI de saída.

    Atributos
    ---------
    name : nome do KPI como ele aparece no cockpit (ex.: "lead_time").
           A coluna efetivamente lida do dataset agregado é
           f"{name}{TARGET_SUFFIX}" — ver _resolve_target_column().
    method : "standard"     — padronização z = (y - média)/desvio
             "log_standard" — log1p e SÓ ENTÃO padronização; para KPIs
                              fortemente enviesados à direita (filas perto
                              da saturação). Escolha explícita, nunca
                              automática.
             "none"         — passthrough real, para medir na monografia
                              o efeito de transformar o alvo.
    """
    name: str
    method: str = "standard"

    def __post_init__(self):
        if self.method not in TARGET_METHODS:
            raise ValueError(
                f"'{self.name}': método de transformação de alvo desconhecido "
                f"{self.method!r}. Use um de {TARGET_METHODS}."
            )


@dataclass
class _FittedTarget:
    """Estatísticas aprendidas para UM KPI — ajustadas SÓ no treino.

    Guardadas explicitamente (em vez de um StandardScaler opaco) porque a
    ausência de vazamento é testável justamente comparando `mean_`/`scale_`
    com a média/desvio calculados à mão sobre train_df.
    """
    method: str
    mean_: float = 0.0
    scale_: float = 1.0
    #: quantos valores do treino entraram no ajuste (diagnóstico)
    n_fit_: int = 0

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return y
        t = np.log1p(y) if self.method == "log_standard" else y
        return (t - self.mean_) / self.scale_

    def inverse_transform(self, y_t: np.ndarray) -> np.ndarray:
        # ORDEM CRÍTICA: desfaz a padronização PRIMEIRO, o log1p DEPOIS.
        # Trocar a ordem é bug silencioso — as previsões saem em escala
        # errada sem levantar exceção nenhuma.
        if self.method == "none":
            return y_t
        t = y_t * self.scale_ + self.mean_
        return np.expm1(t) if self.method == "log_standard" else t


def _fit_target(method: str, y: np.ndarray, kpi: str) -> _FittedTarget:
    """Ajusta as estatísticas de um KPI. Recebe SÓ valores do treino."""
    finitos = y[np.isfinite(y)]
    if finitos.size == 0:
        raise ValueError(
            f"KPI '{kpi}': nenhum valor finito no treino — não dá para ajustar "
            "a transformação de alvo."
        )
    if finitos.size < y.size:
        logger.warning(
            "KPI '%s': %d de %d valores não finitos no treino foram ignorados "
            "no ajuste da transformação (as linhas continuam no DataFrame).",
            kpi, y.size - finitos.size, y.size,
        )

    if method == "none":
        return _FittedTarget(method="none", n_fit_=int(finitos.size))

    if method == "log_standard":
        if np.any(finitos <= -1.0):
            raise ValueError(
                f"KPI '{kpi}': method='log_standard' exige valores > -1 "
                f"(log1p), mas o mínimo no treino é {finitos.min()}. "
                "Use method='standard' para este KPI."
            )
        finitos = np.log1p(finitos)

    mean = float(finitos.mean())
    # ddof=0 (desvio populacional), igual ao StandardScaler do sklearn.
    std = float(finitos.std(ddof=0))
    if std == 0.0:
        logger.warning(
            "KPI '%s': desvio zero no treino (todos os pontos com o mesmo "
            "valor). Usando scale_=1.0 para não dividir por zero — o alvo fica "
            "constante e o metamodelo não tem o que aprender.", kpi,
        )
        std = 1.0
    return _FittedTarget(
        method=method, mean_=mean, scale_=std, n_fit_=int(finitos.size)
    )


# =====================================================================
# 3. O pipeline
# =====================================================================

@dataclass
class PreprocessingPipeline:
    """Prepara o dataset agregado para a Fase 3.

    Uso canônico — a ORDEM importa (ver docstring do módulo):

        pipe = PreprocessingPipeline(param_specs, target_specs)
        train_df, test_df = pipe.split(dataset, test_size=0.2, seed=42)
        pipe.fit(train_df)                       # SÓ o treino

        X_train = pipe.transform_X(train_df)
        X_test  = pipe.transform_X(test_df)
        y_train = pipe.transform_y(train_df, "lead_time")
        y_test  = pipe.transform_y(test_df, "lead_time")

        # depois de o metamodelo prever, para reportar em minutos/peças:
        y_pred = pipe.inverse_transform_y(y_pred_t, "lead_time")

    UM MODELO POR KPI: X é compartilhado entre todos os KPIs, y não — por
    isso transform_y/inverse_transform_y recebem o nome do KPI.
    """

    param_specs: list[ParameterSpec]
    target_specs: list[TargetTransformSpec]

    input_transformer_: ColumnTransformer = field(init=False, repr=False, default=None)
    targets_: dict[str, _FittedTarget] = field(init=False, repr=False, default_factory=dict)
    fitted_: bool = field(init=False, default=False)

    def __post_init__(self):
        if not self.target_specs:
            raise ValueError("target_specs vazio: não há y para preparar.")
        nomes = [t.name for t in self.target_specs]
        duplicados = sorted({n for n in nomes if nomes.count(n) > 1})
        if duplicados:
            raise ValueError(f"target_specs com KPIs repetidos: {duplicados}")

        # Construído já no __init__ de propósito: o transformador de X não
        # depende de dado nenhum, então existir antes de fit() é a própria
        # demonstração de que ele é determinístico.
        self.input_transformer_ = build_input_transformer(self.param_specs)

    # -----------------------------------------------------------------
    # split — ANTES de fit()
    # -----------------------------------------------------------------

    def split(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide o dataset agregado em treino/teste. Chamar ANTES de fit().

        Devolve DataFrames COMPLETOS (não arrays): as colunas que este
        módulo não usa — point_id, {kpi}_std, {kpi}_n, {kpi}_sem e
        principalmente {kpi}_sample_weight — continuam disponíveis para a
        Fase 3 (ex.: passar sample_weight ao fit de um metamodelo
        ponderado). Este módulo não gerencia sample_weight; só não o
        destrói.

        Um train_test_split comum basta porque o dataset agregado tem UMA
        linha por ponto do DoE (as replicações já foram consolidadas por
        aggregate_replications) — não há risco de o mesmo ponto cair dos
        dois lados.

        NOTA PARA O FUTURO (não implementar agora): se algum dia o pipeline
        passar a treinar sobre os dados BRUTOS por replicação, o split
        precisará ser AGRUPADO por `point_id` (GroupShuffleSplit /
        GroupKFold com groups=raw["point_id"]). Sem isso, réplicas do mesmo
        ponto cairiam em lados opostos e o teste ficaria contaminado — o
        metamodelo veria, no treino, praticamente a mesma linha que precisa
        prever no teste.
        """
        if len(dataset) < 2:
            raise ValueError(
                f"dataset com {len(dataset)} linha(s): não há como dividir em "
                "treino e teste."
            )

        train_df, test_df = train_test_split(
            dataset, test_size=test_size, random_state=seed, shuffle=True,
        )
        logger.info(
            "split: %d pontos -> %d treino / %d teste (test_size=%s, seed=%s).",
            len(dataset), len(train_df), len(test_df), test_size, seed,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # -----------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "PreprocessingPipeline":
        """Ajusta os transformadores.

        O de X é determinístico (vem dos specs, não do dado): o fit aqui só
        registra os nomes das colunas de entrada — passar train_df, o
        dataset inteiro ou uma única linha produz o MESMO transformador.

        O de y aprende de verdade (média/desvio de cada KPI) — e é por isso
        que só pode ver train_df. Passar o dataset completo aqui é vazamento
        e invalida as métricas da Fase 4.
        """
        self._check_input_columns(train_df)

        # fit no transformador de X: no-op estatístico, obrigatório só para
        # o sklearn considerar o ColumnTransformer ajustado.
        self.input_transformer_.fit(train_df)

        self.targets_ = {}
        for tspec in self.target_specs:
            col = self._resolve_target_column(train_df, tspec.name)
            y = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype="float64")
            self.targets_[tspec.name] = _fit_target(tspec.method, y, tspec.name)

        self.fitted_ = True
        logger.info(
            "fit: %d linha(s) de treino; X -> %d colunas; alvos ajustados: %s.",
            len(train_df), len(self.get_feature_names()),
            ", ".join(f"{k}({v.method})" for k, v in self.targets_.items()),
        )
        return self

    # -----------------------------------------------------------------
    # transform
    # -----------------------------------------------------------------

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz de entrada codificada, na ordem de get_feature_names().

        Colunas que não são parâmetro declarado são descartadas
        (remainder="drop") — o dataset agregado traz point_id e as colunas
        _mean/_std/_n/_sem/_sample_weight por KPI, e nenhuma delas entra
        em X.
        """
        self._require_fitted("transform_X")
        self._check_input_columns(df)
        return np.asarray(self.input_transformer_.transform(df), dtype="float64")

    def transform_y(self, df: pd.DataFrame, kpi: str) -> np.ndarray:
        """Vetor-alvo de um KPI, transformado.

        ATENÇÃO AO NOME DA COLUNA: o alvo é f"{kpi}{TARGET_SUFFIX}" (ex.:
        "lead_time_mean"), a média das replicações produzida por
        aggregate_replications() — não a coluna "lead_time" pura, que só
        existe nos dados brutos por replicação.
        """
        self._require_fitted("transform_y")
        target = self._get_target(kpi)
        col = self._resolve_target_column(df, kpi)
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        return target.transform(y)

    def inverse_transform_y(self, y_t: np.ndarray, kpi: str) -> np.ndarray:
        """Devolve previsões do metamodelo à escala ORIGINAL do KPI.

        Necessário para reportar RMSE/MAE em unidades interpretáveis
        (minutos, peças) na Fase 4, e para o XAI da Fase 5 falar da
        grandeza real em vez de z-scores.
        """
        self._require_fitted("inverse_transform_y")
        target = self._get_target(kpi)
        return target.inverse_transform(np.asarray(y_t, dtype="float64"))

    def transform_sample_weight(self, df: pd.DataFrame, kpi: str) -> np.ndarray:
        """Peso amostral CONSISTENTE com a escala em que o alvo é treinado.

        POR QUE ISTO EXISTE — é a armadilha mais cara deste módulo.

        replication.aggregate_replications() calcula o peso como 1/variância
        das replicações, medida na escala BRUTA do KPI. Mas o metamodelo não
        treina na escala bruta: treina no que transform_y() devolve. Quando o
        KPI usa method="log_standard", passar o peso bruto direto ao `fit`
        mistura duas escalas — corrige-se uma heterocedasticidade que a
        própria transformação logarítmica já removeu.

        O estrago é grande e silencioso. Medido no cenário do notebook
        (40 pontos, tempo de fila entre 1,2 e 127 min): a variância bruta
        varia por um fator de 3,1e5 entre o ponto mais estável e o mais
        ruidoso, e ponderar por 1/variância colapsa o tamanho efetivo de
        amostra (Kish) de 40 para 5,1 pontos — o ajuste passa a ser
        determinado por meia dúzia de pontos rápidos. Um Random Forest que
        alcança R²=0,75 no teste SEM peso cai para R²=-0,71 COM o peso bruto.
        Nada falha: sai um modelo, com métrica ruim que se confundiria com
        "esta família não serve para o problema".

        COMO O PESO É CONVERTIDO

        - "none" e "standard": a transformação é afim, então ela só reescala
          a variância por uma constante global. Os pesos RELATIVOS entre
          pontos não mudam, e nada precisa ser corrigido.

        - "log_standard": a transformação é não linear, e a conversão usa o
          método delta — Var(log1p(Y)) ≈ Var(Y)/(1+E[Y])² —, que em termos
          de peso vira

              peso_transformado ≈ peso_bruto · (1 + média_do_ponto)²

          Verificado contra a variância empírica do log no mesmo cenário:
          correlação 0,993, erro relativo mediano de 3,9%, e o tamanho
          efetivo de amostra passa de 5,1 (peso bruto) para 16,3 — contra
          16,9 do peso exato calculado sobre as replicações individuais.

        LIMITE DA APROXIMAÇÃO: o método delta é um truncamento de primeira
        ordem e degrada quando o desvio do ponto é grande em relação à sua
        média (coeficiente de variação alto). Para o peso — que é uma
        ponderação relativa, não uma estimativa a ser reportada — essa
        precisão é suficiente. Se algum dia for preciso o valor exato, o
        caminho é calcular a variância do log a partir das replicações
        individuais, antes da agregação.

        Devolve NaN onde o peso de origem é NaN (variância nula, isto é,
        todas as replicações do ponto idênticas). Tratar esses casos é de
        quem chama: este módulo não inventa peso para eles.
        """
        self._require_fitted("transform_sample_weight")
        target = self._get_target(kpi)

        coluna_peso = f"{kpi}{WEIGHT_SUFFIX}"
        if coluna_peso not in df.columns:
            raise ValueError(
                f"KPI '{kpi}': coluna '{coluna_peso}' ausente. Esperado o dataset "
                "agregado de aggregate_replications(), que a produz."
            )

        peso = pd.to_numeric(df[coluna_peso], errors="coerce").to_numpy(dtype="float64")

        if target.method == "log_standard":
            media = pd.to_numeric(
                df[self._resolve_target_column(df, kpi)], errors="coerce"
            ).to_numpy(dtype="float64")
            peso = peso * (1.0 + media) ** 2

        # Fator global (não altera pesos relativos); mantém o peso na unidade
        # do alvo transformado, que é o que um estimador ponderado espera.
        peso = peso * target.scale_ ** 2

        self._avisar_peso_degenerado(peso, kpi)
        return peso

    @staticmethod
    def _avisar_peso_degenerado(peso: np.ndarray, kpi: str) -> None:
        """Avisa quando a ponderação concentra o ajuste em poucos pontos.

        Usa o tamanho efetivo de amostra de Kish, n_ef = (Σw)²/Σw², que é
        quantos pontos de peso igual produziriam a mesma precisão. Um n_ef
        muito abaixo do número de pontos significa que o metamodelo está
        sendo ajustado por uma fração da amostra — situação que não levanta
        exceção e que, sem este aviso, apareceria só como métrica ruim no
        fim da Fase 4, quando já é difícil rastrear a causa.
        """
        finitos = peso[np.isfinite(peso) & (peso > 0)]
        if finitos.size < 2:
            return

        n_efetivo = float(finitos.sum() ** 2 / np.sum(finitos ** 2))
        if n_efetivo < 0.25 * finitos.size:
            logger.warning(
                "KPI '%s': o peso amostral é muito desigual — tamanho efetivo "
                "de amostra (Kish) = %.1f de %d pontos, ou seja, o ajuste será "
                "dominado por uma fração da amostra. Verifique se o método de "
                "transformação do alvo é adequado ao KPI, e considere treinar "
                "também SEM peso para comparar.",
                kpi, n_efetivo, finitos.size,
            )

    # -----------------------------------------------------------------
    # nomes das features
    # -----------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Nomes das colunas DEPOIS do encoding, na ordem de transform_X().

        One-hot expande uma coluna categórica em várias, então estes nomes
        não são os nomes dos parâmetros. A Fase 5 (SHAP/PDP) depende disto
        para rotular as explicações corretamente.

        Limpeza de prefixo: o ColumnTransformer devolve nomes prefixados
        pelo nome do transformador ("routing__routing_FIFO"). Como aqui o
        transformador tem o mesmo nome da coluna, o prefixo é sempre
        redundante — removê-lo dá "routing_FIFO" e "arrival_rate", que é o
        que fica legível num gráfico de SHAP.
        """
        self._require_fitted("get_feature_names")
        brutos = list(self.input_transformer_.get_feature_names_out())
        prefixos = [f"{s.name}__" for s in self.param_specs]
        return [self._strip_prefix(nome, prefixos) for nome in brutos]

    @staticmethod
    def _strip_prefix(nome: str, prefixos: list[str]) -> str:
        """Remove o prefixo do transformador casando contra os nomes de
        parâmetro CONHECIDOS, em vez de dar split no primeiro "__".

        Dá no mesmo resultado — build_input_transformer() já garante que
        nenhum nome de parâmetro contém "__" —, mas não depende dessa
        garantia para estar correto: se a validação mudar, aqui continua
        removendo só o prefixo que de fato existe. Do mais longo para o
        mais curto, para o caso de um nome ser prefixo de outro.
        """
        for prefixo in sorted(prefixos, key=len, reverse=True):
            if nome.startswith(prefixo):
                resto = nome[len(prefixo):]
                return resto or nome
        return nome

    # -----------------------------------------------------------------
    # persistência
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serializa o pipeline AJUSTADO com joblib.

        Persistir junto com o modelo em runs/<run_id>/ é o que garante que
        qualquer previsão futura passe exatamente pela mesma transformação —
        inclusive a inversa, sem a qual as previsões voltam em z-score.
        """
        self._require_fitted("save")
        joblib.dump(self, path)
        logger.info("PreprocessingPipeline salvo em %s.", path)

    @classmethod
    def load(cls, path: str) -> "PreprocessingPipeline":
        """Recarrega um pipeline salvo por save()."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"{path} não contém um {cls.__name__} (veio {type(obj).__name__})."
            )
        return obj

    # -----------------------------------------------------------------
    # internos
    # -----------------------------------------------------------------

    def _require_fitted(self, metodo: str) -> None:
        if not self.fitted_:
            raise RuntimeError(
                f"{metodo}() antes de fit(). A ordem é split() -> fit(train_df) "
                "-> transform_*(): ajustar depois de transformar, ou ajustar no "
                "dataset completo, vaza informação do teste para o treino."
            )

    def _get_target(self, kpi: str) -> _FittedTarget:
        try:
            return self.targets_[kpi]
        except KeyError:
            raise KeyError(
                f"KPI {kpi!r} não está em target_specs "
                f"({sorted(self.targets_)}). Um metamodelo por KPI: declare o "
                "KPI em target_specs para ele ter transformação própria."
            ) from None

    def _check_input_columns(self, df: pd.DataFrame) -> None:
        faltando = [s.name for s in self.param_specs if s.name not in df.columns]
        if faltando:
            raise ValueError(
                f"Colunas de parâmetro ausentes no DataFrame: {faltando}. "
                "Esperado o dataset agregado de aggregate_replications(), que "
                "carrega uma coluna por ParameterSpec."
            )

    @staticmethod
    def _resolve_target_column(df: pd.DataFrame, kpi: str) -> str:
        """Descobre qual coluna carrega o alvo do KPI.

        Caminho normal: f"{kpi}_mean", a média das replicações que
        aggregate_replications() produz — é ESSE o alvo do metamodelo.

        Fallback documentado: se a coluna _mean não existir mas existir uma
        coluna com o nome puro do KPI, ela é usada. Serve a datasets não
        agregados (exploração, teste, ou uma resposta determinística sem
        replicação). Fica em INFO no log para não passar despercebido.
        """
        alvo = f"{kpi}{TARGET_SUFFIX}"
        if alvo in df.columns:
            return alvo
        if kpi in df.columns:
            logger.info(
                "KPI '%s': coluna '%s' não encontrada; usando a coluna '%s'. "
                "Isso só é esperado em dataset NÃO agregado — o alvo canônico "
                "é a média das replicações.", kpi, alvo, kpi,
            )
            return kpi
        raise ValueError(
            f"KPI '{kpi}': nem a coluna '{alvo}' nem '{kpi}' existem no "
            f"DataFrame. Colunas disponíveis: {sorted(df.columns)}."
        )
