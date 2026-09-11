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
from smt.sampling_methods import LHS


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

        lhs         método principal — amostragem estratificada
        monte_carlo BASELINE de comparação — amostragem aleatória simples
        sobol       alternativa para alta dimensionalidade; só aceita espaço
                    numérico (ver a validação abaixo)

        CUIDADO COM `n_points` NO SOBOL: as propriedades de balanceamento de
        uma sequência de Sobol valem para n potência de 2 (64, 128, 256...).
        Com outro n o scipy emite UserWarning e a sequência perde parte da
        uniformidade que justifica usá-la. Não é erro — mas escolher
        n_points=100 para Sobol desperdiça a vantagem do método.

    criterion : critério de otimização do LHS (default "ese", o mesmo usado
                por Saves et al. 2025). Só tem efeito com method="lhs";
                aceita também "center", "maximin", "centermaximin", "corr".

    Retorna
    -------
    pd.DataFrame com `n_points` linhas, uma coluna por parâmetro,
    já no dtype correto (int64/float64/category).
    """
    if method not in ("lhs", "monte_carlo", "sobol"):
        raise ValueError(f"Método de DoE desconhecido: {method!r}")

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

    unit_cube = _sample_unit_cube(
        method=method, n_points=n_points, n_dims=len(specs), seed=seed,
        criterion=criterion,
    )
    df = _decode_unit_cube(unit_cube, specs)
    return _cast_dtypes(df, specs)


# ---------------------------------------------------------------------
# 4. Amostragem no cubo unitário e decodificação para o espaço declarado
# ---------------------------------------------------------------------
#
# POR QUE OS TRÊS MÉTODOS PASSAM PELO CUBO UNITÁRIO
#
# A versão anterior delegava LHS e Monte Carlo ao invólucro
# MixedIntegerSamplingMethod do SMT, e isso produzia um defeito silencioso:
# os dois métodos devolviam EXATAMENTE o mesmo plano. A causa está na fonte
# do SMT (2.14.1) — MixedIntegerSamplingMethod._compute() chama
# `self._design_space.sample_valid_x(...)` e nunca usa o
# `self._sampling_method` que construiu a partir da classe recebida; a
# classe do amostrador só é usada pelo método expand_lhs(), que não
# chamamos. E DesignSpace._sample_valid_x() fixa internamente
# LHS(criterion="ese"). Consequência: pedir "monte_carlo" devolvia LHS, e o
# parâmetro `criterion` era inerte.
#
# Isso não é detalhe de implementação: a Tabela de métodos de amostragem da
# monografia posiciona o Monte Carlo como o BASELINE contra o qual LHS e
# Sobol são comparados. Sem um Monte Carlo de verdade, a comparação de
# preenchimento do espaço — que é justamente o critério de verificação do
# requisito funcional de geração de dados via DoE — não pode ser produzida.
#
# A correção amostra no cubo unitário [0,1)^d com o método pedido e decodifica
# para o espaço declarado num caminho único, compartilhado pelos três métodos.
# Ganhos: (a) Monte Carlo é Monte Carlo; (b) `criterion` volta a ter efeito;
# (c) os três métodos ficam comparáveis porque diferem SÓ no preenchimento do
# cubo, e não também na forma de decodificar.


def _sample_unit_cube(
    method: str, n_points: int, n_dims: int, seed: int, criterion: str
) -> np.ndarray:
    """Preenche [0,1)^n_dims com `n_points` pontos, pelo método pedido."""
    if method == "lhs":
        # LHS do SMT direto (sem o invólucro mixed-integer), sobre o cubo
        # unitário. Aqui `criterion` chega de fato ao otimizador.
        sampler = LHS(
            xlimits=np.tile([0.0, 1.0], (n_dims, 1)), criterion=criterion, seed=seed
        )
        return np.asarray(sampler(n_points), dtype="float64")

    if method == "monte_carlo":
        # Amostragem aleatória simples — o BASELINE. Gerador próprio do numpy
        # em vez do sampler Random do SMT: é uma linha, não depende do
        # comportamento do invólucro e a semente fica sob nosso controle.
        return np.random.default_rng(seed).random((n_points, n_dims))

    # sobol
    from scipy.stats import qmc

    return np.asarray(
        qmc.Sobol(d=n_dims, seed=seed).random(n_points), dtype="float64"
    )


def _decode_unit_cube(unit_cube: np.ndarray, specs: list[ParameterSpec]) -> pd.DataFrame:
    """Mapeia cada coluna de [0,1) para o domínio declarado no spec.

    Fatores discretos usam `floor(u * n_niveis)` em vez de arredondar um
    valor contínuo. A diferença importa: arredondar low + u*(high-low) dá aos
    níveis EXTREMOS metade da probabilidade dos internos (as caixas das
    pontas têm metade da largura), o que enviesaria o design contra as bordas
    do espaço — exatamente a região que um DoE precisa cobrir bem. Com floor,
    todos os níveis são equiprováveis.
    """
    colunas = {}
    for j, spec in enumerate(specs):
        u = np.clip(unit_cube[:, j], 0.0, np.nextafter(1.0, 0.0))

        if spec.type == "float":
            colunas[spec.name] = spec.low + u * (spec.high - spec.low)

        elif spec.type == "integer":
            n_niveis = int(spec.high) - int(spec.low) + 1
            colunas[spec.name] = int(spec.low) + np.floor(u * n_niveis).astype("int64")

        elif spec.type in ("categorical", "ordinal"):
            idx = np.floor(u * len(spec.levels)).astype("int64")
            colunas[spec.name] = [spec.levels[i] for i in idx]

        elif spec.type == "binary":
            idx = np.floor(u * 2).astype("int64")
            # ordered=True  -> IntegerVariable(0,1): inteiro 0/1
            # ordered=False -> CategoricalVariable(["0","1"]): STRING "0"/"1"
            #   (ver _to_smt_variable; o pré-processamento conta com isso)
            colunas[spec.name] = idx if spec.ordered else [str(i) for i in idx]

        else:
            raise ValueError(f"Tipo de parâmetro desconhecido: {spec.type!r}")

    return pd.DataFrame(colunas)


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