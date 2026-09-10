"""
Fase 1 do pipeline — Plano de Experimentos (DoE).

Responsabilidade única deste módulo: dado um espaço de parâmetros
(qualquer combinação de float, integer, categorical, binary, ordinal) e
um número de pontos desejado, gerar um DataFrame com esse número de
linhas cobrindo o espaço — de forma one-shot (uma única chamada,
sem realimentação), conforme decisão da monografia.

Este módulo NÃO sabe nada sobre replicações, warm-up ou o simulador.
Essas são responsabilidades de replication.py e de sim_interface/.
O DataFrame gerado aqui alimenta, via replication.build_execution_plan(),
o cockpit Excel trocado com o ProModel (sim_interface/excel_bridge.py):
cada coluna criada aqui vira uma coluna do bloco de ENTRADA da planilha.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from smt.design_space import (
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    CategoricalVariable,
    OrdinalVariable,
)
from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.sampling_methods import LHS, Random


# ---------------------------------------------------------------------
# 1. Especificação de cada parâmetro (o que vem do YAML do cenário)
# ---------------------------------------------------------------------

@dataclass
class ParameterSpec:
    """Descreve uma coluna do espaço de parâmetros de entrada.

    Atributos
    ---------
    name : nome da coluna (vira nome de coluna no DataFrame final)
    type : "float" | "integer" | "categorical" | "binary" | "ordinal"
    low, high : limites (obrigatórios para "float" e "integer")
    levels : lista de valores possíveis (obrigatório para "categorical" e
             "ordinal"). Para "ordinal", a ORDEM da lista importa: define
             a ordem do atributo — use quando os níveis têm ordem natural
             mas espaçamento não necessariamente uniforme (ex.: prioridade
             ["baixa", "media", "alta"] ou tamanho de lote [10, 50, 200]).
             Diferente de "categorical" (nominal, sem ordem) e de "integer"
             (numérico, espaçamento uniforme).
    ordered : só relevante para "binary" — ver nota abaixo
    """
    name: str
    type: str
    low: float | None = None
    high: float | None = None
    levels: list | None = None
    ordered: bool = True

    def __post_init__(self):
        if self.type in ("float", "integer") and (self.low is None or self.high is None):
            raise ValueError(f"'{self.name}': tipo '{self.type}' exige 'low' e 'high'.")
        if self.type in ("categorical", "ordinal") and not self.levels:
            raise ValueError(f"'{self.name}': tipo '{self.type}' exige 'levels'.")


# ---------------------------------------------------------------------
# 2. Tradução ParameterSpec -> variável do SMT
# ---------------------------------------------------------------------

def _to_smt_variable(spec: ParameterSpec):
    if spec.type == "float":
        return FloatVariable(spec.low, spec.high)
    if spec.type == "integer":
        return IntegerVariable(spec.low, spec.high)
    if spec.type == "categorical":
        return CategoricalVariable(spec.levels)
    if spec.type == "ordinal":
        # OrdinalVariable preserva a ordem de spec.levels (diferente de
        # CategoricalVariable, que trata os níveis como nominais/sem ordem).
        return OrdinalVariable(spec.levels)
    if spec.type == "binary":
        # ordered=True  -> IntegerVariable(0,1): existe noção de "entre 0 e 1"
        #                  (ex.: liga/desliga um recurso, onde interpolar faz sentido)
        # ordered=False -> CategoricalVariable(["0","1"]): troca qualitativa de modo
        #                  (ex.: alternar entre duas regras de despacho distintas)
        return IntegerVariable(0, 1) if spec.ordered else CategoricalVariable(["0", "1"])
    raise ValueError(f"Tipo de parâmetro desconhecido: {spec.type!r}")


def build_design_space(specs: list[ParameterSpec]) -> tuple[DesignSpace, list[str]]:
    """Monta o DesignSpace do SMT a partir da lista de specs.
    Retorna também a ordem dos nomes, porque o SMT trabalha com
    arrays posicionais (numpy), não com nomes de coluna."""
    variables = [_to_smt_variable(s) for s in specs]
    names = [s.name for s in specs]
    return DesignSpace(variables), names


# ---------------------------------------------------------------------
# 3. Geração do plano (a função que você realmente chama)
# ---------------------------------------------------------------------

def generate_design(
    specs: list[ParameterSpec],
    n_points: int,
    seed: int,
    method: str = "lhs",
    criterion: str = "ese",
) -> pd.DataFrame:
    """Gera o plano de experimentos ONE-SHOT.

    Parâmetros
    ----------
    specs : lista de ParameterSpec — o espaço de entrada completo
    n_points : quantos pontos (linhas) o design deve ter
    seed : semente para reprodutibilidade
    method : "lhs" (padrão) | "monte_carlo" | "sobol"
    criterion : critério de otimização do LHS (default "ese",
                o mesmo usado por Saves et al. 2025)

    Retorna
    -------
    pd.DataFrame com `n_points` linhas, uma coluna por parâmetro,
    já no dtype correto (int64/float64/category).
    """
    has_discrete_labeled = any(
        s.type in ("categorical", "ordinal") or (s.type == "binary" and not s.ordered)
        for s in specs
    )

    if method == "sobol" and has_discrete_labeled:
        raise ValueError(
            "Sobol não tem definição nativa para variáveis categóricas nominais "
            "nem ordinais (a implementação de Sobol aqui é contínua, via scipy, "
            "e não conhece a lista de níveis de 'ordinal'). "
            "Use method='lhs' para este espaço de parâmetros."
        )

    design_space, names = build_design_space(specs)

    if method == "lhs":
        sampler = MixedIntegerSamplingMethod(
            LHS, design_space, criterion=criterion, seed=seed
        )
        raw = sampler(n_points)
        df = _decode_to_dataframe(raw, design_space, names)

    elif method == "monte_carlo":
        sampler = MixedIntegerSamplingMethod(Random, design_space, seed=seed)
        raw = sampler(n_points)
        df = _decode_to_dataframe(raw, design_space, names)

    elif method == "sobol":
        # Sobol só faz sentido aqui em espaços contínuos/inteiros (sem categórico
        # nominal, já validado acima). Implementado via scipy diretamente porque
        # o SMT não expõe Sobol como sampler de mixed-integer.
        from scipy.stats import qmc
        sobol_engine = qmc.Sobol(d=len(specs), seed=seed)
        unit_cube = sobol_engine.random(n_points)
        raw = np.zeros_like(unit_cube)
        for j, s in enumerate(specs):
            raw[:, j] = s.low + unit_cube[:, j] * (s.high - s.low)
        df = pd.DataFrame(raw, columns=names)

    else:
        raise ValueError(f"Método de DoE desconhecido: {method!r}")

    return _cast_dtypes(df, specs)


# ---------------------------------------------------------------------
# 4. Decodificação (LHS/Random do SMT devolvem códigos numéricos para
#    variáveis categóricas — ex.: "SPT" vira 1.0 — decode_values traduz
#    de volta para o rótulo original antes de virar DataFrame)
# ---------------------------------------------------------------------

def _decode_to_dataframe(raw: np.ndarray, design_space: DesignSpace, names: list[str]) -> pd.DataFrame:
    decoded_rows = design_space.decode_values(raw)
    return pd.DataFrame(decoded_rows, columns=names)


# ---------------------------------------------------------------------
# 5. Ajuste final de tipos
# ---------------------------------------------------------------------

def _cast_dtypes(df: pd.DataFrame, specs: list[ParameterSpec]) -> pd.DataFrame:
    """Sem isso, colunas inteiras saem como float64 (1.0, 2.0, 3.0)
    e colunas categóricas saem com dtype 'object' misto."""
    for s in specs:
        if s.type == "integer" or (s.type == "binary" and s.ordered):
            df[s.name] = df[s.name].round().astype("int64")
        elif s.type == "float":
            df[s.name] = df[s.name].astype("float64")
        elif s.type == "categorical" or (s.type == "binary" and not s.ordered):
            df[s.name] = df[s.name].astype("category")
        elif s.type == "ordinal":
            # category ORDENADA (ordered=True), com a ordem de s.levels —
            # preserva a semântica de ordinal downstream (ex.: comparações
            # <, <=, .cat.codes monotônico, plots/XAI que respeitam a ordem).
            df[s.name] = pd.Categorical(
                df[s.name], categories=s.levels, ordered=True
            )
    return df