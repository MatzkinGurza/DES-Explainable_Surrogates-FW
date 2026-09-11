"""Testes da geração do plano de experimentos (Fase 1).

O foco desta suíte é o contrato dos TRÊS MÉTODOS de amostragem. Ela existe
porque a versão anterior tinha um defeito silencioso: "monte_carlo" devolvia
exatamente o mesmo plano que "lhs", e `criterion` não tinha efeito nenhum —
nada falhava, o design saía, e a comparação de preenchimento do espaço
prevista na monografia simplesmente não era possível. Defeito que não quebra
nada só é pego por teste que o procure de propósito.
"""

import numpy as np
import pandas as pd
import pytest

from sed_meta.doe.parameter_space import ParameterSpec, generate_design

METODOS = ["lhs", "monte_carlo", "sobol"]


@pytest.fixture
def specs_num():
    """Espaço só numérico — o único que os três métodos aceitam."""
    return [
        ParameterSpec("taxa_chegada", "float", low=5.0, high=20.0),
        ParameterSpec("num_operadores", "integer", low=1, high=4),
    ]


@pytest.fixture
def specs_mistos():
    """Os cinco tipos, incluindo binário ordenado e não-ordenado."""
    return [
        ParameterSpec("taxa_chegada", "float", low=5.0, high=20.0),
        ParameterSpec("num_operadores", "integer", low=1, high=4),
        ParameterSpec("regra_despacho", "categorical", levels=["FIFO", "SPT", "EDD"]),
        ParameterSpec("prioridade", "ordinal", levels=["baixa", "media", "alta"]),
        ParameterSpec("turno_extra", "binary", ordered=True),
        ParameterSpec("modo_setup", "binary", ordered=False),
    ]


# ---------------------------------------------------------------------
# O defeito que esta suíte existe para impedir
# ---------------------------------------------------------------------

def test_monte_carlo_nao_e_igual_ao_lhs(specs_mistos):
    """O Monte Carlo é o BASELINE contra o qual LHS e Sobol são comparados
    na monografia. Se ele devolver o mesmo plano que o LHS, a comparação de
    preenchimento do espaço deixa de existir — sem que nada falhe."""
    lhs = generate_design(specs_mistos, n_points=100, seed=7, method="lhs")
    mc = generate_design(specs_mistos, n_points=100, seed=7, method="monte_carlo")

    assert not np.allclose(lhs["taxa_chegada"], mc["taxa_chegada"])


def test_criterion_altera_o_design(specs_num):
    """`criterion` é opção declarada de generate_design. Se for inerte, quem
    chama acredita estar ajustando a otimização do LHS e não está."""
    ese = generate_design(specs_num, n_points=60, seed=7, method="lhs", criterion="ese")
    center = generate_design(
        specs_num, n_points=60, seed=7, method="lhs", criterion="center"
    )

    assert not np.allclose(ese["taxa_chegada"], center["taxa_chegada"])


def test_lhs_preenche_melhor_que_monte_carlo(specs_num):
    """A premissa que justifica o LHS como método principal na monografia:
    cobertura marginal mais uniforme que a amostragem aleatória.

    Mede-se pela dispersão dos espaçamentos entre pontos consecutivos na
    marginal — num LHS os pontos caem um por estrato, então os espaçamentos
    são regulares; na amostragem aleatória eles se aglomeram e deixam vazios.
    Média sobre várias sementes porque uma única amostra aleatória pode, por
    acaso, sair bem distribuída.
    """
    def dispersao(metodo, seed):
        d = generate_design(specs_num, n_points=80, seed=seed, method=metodo)
        u = np.sort((d["taxa_chegada"].to_numpy(float) - 5.0) / 15.0)
        return float(np.std(np.diff(u)))

    lhs = np.mean([dispersao("lhs", s) for s in range(8)])
    mc = np.mean([dispersao("monte_carlo", s) for s in range(8)])

    assert lhs < mc, f"LHS deveria cobrir melhor a marginal: {lhs:.5f} vs {mc:.5f}"


# ---------------------------------------------------------------------
# Contrato comum aos três métodos
# ---------------------------------------------------------------------

@pytest.mark.parametrize("metodo", METODOS)
def test_reprodutibilidade(specs_num, metodo):
    """RNF-2: mesma semente, mesmo plano; sementes distintas, planos
    distintos."""
    a = generate_design(specs_num, n_points=50, seed=3, method=metodo)
    b = generate_design(specs_num, n_points=50, seed=3, method=metodo)
    c = generate_design(specs_num, n_points=50, seed=4, method=metodo)

    assert a.equals(b)
    assert not a["taxa_chegada"].equals(c["taxa_chegada"])


@pytest.mark.parametrize("metodo", METODOS)
def test_forma_e_dominio(specs_num, metodo):
    d = generate_design(specs_num, n_points=40, seed=1, method=metodo)

    assert len(d) == 40
    assert list(d.columns) == ["taxa_chegada", "num_operadores"]
    assert d["taxa_chegada"].between(5.0, 20.0).all()
    assert d["num_operadores"].between(1, 4).all()


@pytest.mark.parametrize("metodo", ["lhs", "monte_carlo"])
def test_dtypes_de_todos_os_tipos(specs_mistos, metodo):
    """Os dtypes são contrato com o resto da cadeia: o cockpit os serializa e
    o pré-processamento os codifica."""
    d = generate_design(specs_mistos, n_points=60, seed=1, method=metodo)

    assert d["taxa_chegada"].dtype == "float64"
    assert d["num_operadores"].dtype == "int64"
    assert d["turno_extra"].dtype == "int64"
    assert isinstance(d["regra_despacho"].dtype, pd.CategoricalDtype)

    # ordinal: a ORDEM declarada tem de sobreviver
    prio = d["prioridade"].dtype
    assert isinstance(prio, pd.CategoricalDtype)
    assert prio.ordered is True
    assert list(prio.categories) == ["baixa", "media", "alta"]

    # binário não-ordenado chega como STRING "0"/"1" — o pré-processamento
    # conta com isso (ver preprocessing.pipeline.build_input_transformer)
    assert set(d["modo_setup"].unique()) <= {"0", "1"}
    assert all(isinstance(v, str) for v in d["modo_setup"].unique())


@pytest.mark.parametrize("metodo", ["lhs", "monte_carlo"])
def test_niveis_discretos_sao_equiprovaveis(specs_mistos, metodo):
    """Arredondar um contínuo daria aos níveis das PONTAS metade da
    probabilidade dos internos (caixas de meia largura), enviesando o design
    contra as bordas do espaço — justamente onde um DoE precisa cobrir bem.
    A decodificação usa floor para evitar isso."""
    d = generate_design(specs_mistos, n_points=3000, seed=5, method=metodo)

    for coluna, n_niveis in (("num_operadores", 4), ("regra_despacho", 3)):
        freq = d[coluna].value_counts(normalize=True)
        assert len(freq) == n_niveis
        assert np.allclose(freq.to_numpy(), 1 / n_niveis, atol=0.04), (
            f"{coluna} desbalanceado em {metodo}: {freq.to_dict()}"
        )


# ---------------------------------------------------------------------
# Limites declarados
# ---------------------------------------------------------------------

def test_sobol_recusa_espaco_com_fator_rotulado(specs_mistos):
    """Recusa deliberada: Sobol não tem definição nativa para categórico
    nominal nem ordinal. Melhor estourar do que devolver um design que
    ignora a natureza do fator."""
    with pytest.raises(ValueError, match="Sobol"):
        generate_design(specs_mistos, n_points=30, seed=1, method="sobol")


def test_metodo_desconhecido_estoura(specs_num):
    with pytest.raises(ValueError, match="desconhecido"):
        generate_design(specs_num, n_points=10, seed=1, method="halton")
