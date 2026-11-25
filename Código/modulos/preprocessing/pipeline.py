"""
Pré-processamento: das execuções completas às matrizes X/y do metamodelo.

Regra que organiza o módulo: a transformação das ENTRADAS é determinística,
ancorada nos limites declarados no YAML (nunca nos observados); a das
SAÍDAS é aprendida e SÓ pode ver o treino.

    pipe = PreprocessingPipeline(param_specs, target_specs)
    train_df, test_df = pipe.split(dados, test_size=0.2, seed=0)
    pipe.fit(train_df)                          # só o treino
    X_train, y_train = pipe.transform_X(train_df), pipe.transform_y(train_df, kpi)
    y_pred = pipe.inverse_transform_y(modelo.predict(X_test), kpi)   # em segundos

Entradas: float/integer -> [0, 1] por (x - low) / (high - low);
categorical/binary -> one-hot; ordinal -> posição do nível em [0, 1].

Saídas, uma transformação por indicador (TargetTransformSpec):
    "standard"      z = (y - média) / desvio, estatísticas do treino
    "log_standard"  log1p e depois padronização
    "none"          sem transformação

Split AGRUPADO por point_id: as réplicas de um ponto têm as mesmas entradas,
então um ponto vai inteiro para o treino ou para o teste. Sem isso, o modelo
seria avaliado em entradas que já viu no treino. `test_size` é a fração de
PONTOS.
"""

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from modulos.doe.parameter_space import ParameterSpec

logger = logging.getLogger(__name__)

#: Métodos aceitos em TargetTransformSpec.method.
TARGET_METHODS = ("standard", "log_standard", "none")


# =====================================================================
# 1. Transformação das ENTRADAS (X) — determinística, vinda dos specs
# =====================================================================
#
# POR QUE CLASSES CHAMÁVEIS E NÃO lambdas: joblib serializa o pipeline
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
    """Como transformar UM indicador de saída: "standard", "log_standard" ou "none"."""

    name: str
    method: str = "standard"

    def __post_init__(self):
        if self.method not in TARGET_METHODS:
            raise ValueError(
                f"'{self.name}': método de transformação desconhecido "
                f"{self.method!r}. Use um de {TARGET_METHODS}."
            )


@dataclass
class _FittedTarget:
    """Estatísticas de UM indicador, ajustadas só no treino."""

    method: str
    mean_: float = 0.0
    scale_: float = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return y
        t = np.log1p(y) if self.method == "log_standard" else y
        return (t - self.mean_) / self.scale_

    def inverse_transform(self, y_t: np.ndarray) -> np.ndarray:
        # desfaz a padronização primeiro e o log1p depois
        if self.method == "none":
            return y_t
        t = y_t * self.scale_ + self.mean_
        return np.expm1(t) if self.method == "log_standard" else t


def _fit_target(method: str, y: np.ndarray, kpi: str) -> _FittedTarget:
    """Ajusta as estatísticas de um indicador. Recebe SÓ valores do treino."""
    finitos = y[np.isfinite(y)]
    if finitos.size == 0:
        raise ValueError(f"'{kpi}': nenhum valor finito no treino.")
    if method == "none":
        return _FittedTarget(method="none")
    if method == "log_standard":
        if np.any(finitos <= -1.0):
            raise ValueError(f"'{kpi}': log_standard exige valores > -1 (log1p).")
        finitos = np.log1p(finitos)
    std = float(finitos.std(ddof=0))   # desvio populacional, como o StandardScaler
    if std == 0.0:
        logger.warning("'%s': desvio zero no treino; usando escala 1.", kpi)
        std = 1.0
    return _FittedTarget(method=method, mean_=float(finitos.mean()), scale_=std)


# =====================================================================
# 3. O pipeline
# =====================================================================

@dataclass
class PreprocessingPipeline:
    """Split agrupado, transformação das entradas e dos indicadores.

    Ordem obrigatória: split() -> fit(train_df) -> transform_*().
    X é compartilhado entre os indicadores; y é transformado por indicador.
    """

    param_specs: list[ParameterSpec]
    target_specs: list[TargetTransformSpec]
    input_transformer_: ColumnTransformer = field(init=False, repr=False, default=None)
    targets_: dict[str, _FittedTarget] = field(init=False, repr=False, default_factory=dict)
    fitted_: bool = field(init=False, default=False)

    def __post_init__(self):
        if not self.target_specs:
            raise ValueError("target_specs vazio: não há y para preparar.")
        # o transformador de X não depende de dados: existe antes do fit()
        self.input_transformer_ = build_input_transformer(self.param_specs)

    def split(self, dados: pd.DataFrame, test_size: float = 0.2, seed: int = 0,
              group_col: str = "point_id") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Treino/teste agrupado por ponto: todas as réplicas de um ponto vão
        para o mesmo lado. `test_size` é a fração de PONTOS."""
        separador = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        idx_treino, idx_teste = next(separador.split(dados, groups=dados[group_col]))
        train_df = dados.iloc[idx_treino].reset_index(drop=True)
        test_df = dados.iloc[idx_teste].reset_index(drop=True)
        comuns = set(train_df[group_col]) & set(test_df[group_col])
        assert not comuns, f"{len(comuns)} ponto(s) nos dois lados do split"
        return train_df, test_df

    def fit(self, train_df: pd.DataFrame) -> "PreprocessingPipeline":
        """Ajusta as transformações. Só a de y aprende algo (média e desvio de
        cada indicador), por isso recebe apenas o treino."""
        self._check_input_columns(train_df)
        self.input_transformer_.fit(train_df)
        self.targets_ = {
            t.name: _fit_target(t.method, pd.to_numeric(train_df[t.name]).to_numpy(dtype="float64"), t.name)
            for t in self.target_specs
        }
        self.fitted_ = True
        return self

    def transform_X(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz de entrada codificada, na ordem de get_feature_names()."""
        self._require_fitted("transform_X")
        self._check_input_columns(df)
        return np.asarray(self.input_transformer_.transform(df), dtype="float64")

    def transform_y(self, df: pd.DataFrame, kpi: str) -> np.ndarray:
        """Vetor de um indicador, transformado."""
        self._require_fitted("transform_y")
        y = pd.to_numeric(df[kpi]).to_numpy(dtype="float64")
        return self._get_target(kpi).transform(y)

    def inverse_transform_y(self, y_t: np.ndarray, kpi: str) -> np.ndarray:
        """Leva previsões de volta à escala original do indicador (segundos)."""
        self._require_fitted("inverse_transform_y")
        return self._get_target(kpi).inverse_transform(np.asarray(y_t, dtype="float64"))

    def get_feature_names(self) -> list[str]:
        """Nomes das colunas de X depois da codificação (sem o prefixo do
        ColumnTransformer), usados para rotular o SHAP."""
        self._require_fitted("get_feature_names")
        prefixos = sorted((f"{s.name}__" for s in self.param_specs), key=len, reverse=True)
        nomes = []
        for bruto in self.input_transformer_.get_feature_names_out():
            prefixo = next((p for p in prefixos if bruto.startswith(p)), "")
            nomes.append(bruto[len(prefixo):] or bruto)
        return nomes

    # ---- internos -----------------------------------------------------

    def _require_fitted(self, metodo: str) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{metodo}() antes de fit(). A ordem é split() -> fit(train_df) -> transform_*().")

    def _get_target(self, kpi: str) -> _FittedTarget:
        if kpi not in self.targets_:
            raise KeyError(f"'{kpi}' não está em target_specs ({sorted(self.targets_)}).")
        return self.targets_[kpi]

    def _check_input_columns(self, df: pd.DataFrame) -> None:
        faltando = [s.name for s in self.param_specs if s.name not in df.columns]
        if faltando:
            raise ValueError(f"Colunas de parâmetro ausentes no DataFrame: {faltando}.")
