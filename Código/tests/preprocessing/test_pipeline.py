"""Testes do módulo sed_meta.preprocessing.

O que estes testes existem para provar (na ordem da monografia):

  - X é DETERMINÍSTICO: ajustar em qualquer subconjunto de linhas —
    inclusive uma só — dá exatamente a mesma transformação, porque os
    limites vêm do ParameterSpec e não do dado observado.
  - y é APRENDIDO, e só do treino: as estatísticas guardadas batem com a
    média/desvio calculados à mão sobre train_df, NÃO sobre o dataset
    completo. É isso que mantém honestas as métricas da Fase 4 e a
    comparação contra o teto de ruído.
  - a inversa desfaz a transformação na ordem certa (o bug de trocar
    expm1 com a des-padronização é silencioso: sai número, só que na
    escala errada).

Dados sempre gerados com numpy.random.default_rng(seed) — nada de valor
mágico digitado à mão.
"""

import numpy as np
import pandas as pd
import pytest

from sed_meta.doe.parameter_space import ParameterSpec, generate_design
from sed_meta.doe.replication import (
    ReplicationPolicy,
    aggregate_replications,
    build_execution_plan,
)
from sed_meta.preprocessing import (
    PreprocessingPipeline,
    TargetTransformSpec,
    build_input_transformer,
)

SEED = 20240917
KPIS = ["lead_time", "throughput"]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def specs():
    """Os cinco casos de ParameterSpec que o DoE sabe gerar: float,
    integer, categorical, ordinal, binary ordenado e binary não-ordenado."""
    return [
        ParameterSpec("taxa_chegada", "float", low=0.5, high=4.0),
        ParameterSpec("num_operadores", "integer", low=1, high=4),
        ParameterSpec("regra_despacho", "categorical", levels=["FIFO", "SPT", "EDD"]),
        ParameterSpec("prioridade", "ordinal", levels=["baixa", "media", "alta"]),
        ParameterSpec("turno_extra", "binary", ordered=True),
        ParameterSpec("modo_setup", "binary", ordered=False),
    ]


@pytest.fixture
def target_specs():
    return [
        TargetTransformSpec("lead_time", method="log_standard"),
        TargetTransformSpec("throughput", method="standard"),
    ]


@pytest.fixture
def design(specs):
    return generate_design(specs, n_points=40, seed=SEED, method="lhs")


@pytest.fixture
def dataset(specs, design):
    """Dataset agregado no formato EXATO de aggregate_replications(): uma
    linha por ponto, com _mean/_std/_n/_sem/_sample_weight por KPI.

    Os KPIs sintéticos são gerados a partir dos próprios parâmetros (com
    ruído), para que exista sinal de verdade — e lead_time sai enviesado à
    direita de propósito, que é o caso que motiva o "log_standard".
    """
    return _simular(specs, design, seed=SEED)


def _simular(specs, design, seed, n_replications=5):
    """Roda o caminho real do DoE (plano de execução) e preenche os KPIs
    com uma resposta sintética, devolvendo o dataset agregado."""
    plan = build_execution_plan(design, ReplicationPolicy(n_replications, seed_base=0))
    rng = np.random.default_rng(seed)

    carga = plan["taxa_chegada"].to_numpy(dtype="float64")
    capacidade = plan["num_operadores"].to_numpy(dtype="float64")
    utilizacao = np.clip(carga / (capacidade * 1.4), 0.05, 0.97)

    # fila M/M/1-ish: explode perto da saturação -> cauda longa à direita
    lead_time = 5.0 * utilizacao / (1.0 - utilizacao)
    lead_time = lead_time * rng.lognormal(mean=0.0, sigma=0.15, size=len(plan))
    throughput = 60.0 * utilizacao * capacidade + rng.normal(0.0, 1.5, size=len(plan))

    raw = plan.copy()
    raw["lead_time"] = lead_time
    raw["throughput"] = throughput

    param_cols = [s.name for s in specs]
    return aggregate_replications(raw, param_cols=param_cols, kpi_cols=KPIS)


@pytest.fixture
def fitted(specs, target_specs, dataset):
    """Pipeline no uso canônico: split PRIMEIRO, fit só no treino."""
    pipe = PreprocessingPipeline(specs, target_specs)
    train_df, test_df = pipe.split(dataset, test_size=0.25, seed=7)
    pipe.fit(train_df)
    return pipe, train_df, test_df


# =====================================================================
# 1. Determinismo de X
# =====================================================================

def test_x_invariante_ao_subconjunto_de_ajuste(specs, dataset):
    """Ajustar em subconjuntos diferentes produz transformação IDÊNTICA
    para uma linha comum a ambos — a propriedade central do módulo."""
    referencia = build_input_transformer(specs).fit(dataset).transform(dataset)

    rng = np.random.default_rng(SEED)
    linha_alvo = int(rng.integers(len(dataset)))

    subconjuntos = {
        "uma linha só": dataset.iloc[[linha_alvo]],
        "metade": dataset.iloc[: len(dataset) // 2],
        "amostra aleatória": dataset.sample(8, random_state=1),
    }
    # a linha alvo precisa estar em todos os subconjuntos
    subconjuntos = {
        nome: pd.concat([sub, dataset.iloc[[linha_alvo]]]).drop_duplicates("point_id")
        for nome, sub in subconjuntos.items()
    }

    for nome, sub in subconjuntos.items():
        ct = build_input_transformer(specs).fit(sub)
        saida = ct.transform(dataset.iloc[[linha_alvo]])
        assert np.array_equal(saida, referencia[[linha_alvo]]), (
            f"ajuste em '{nome}' mudou a transformação da linha {linha_alvo}"
        )


def test_x_usa_limites_do_spec_e_nao_os_observados(specs, dataset):
    """Um float no low/high DECLARADO tem de virar exatamente 0.0/1.0,
    mesmo que o dado observado nunca chegue lá."""
    ct = build_input_transformer(specs).fit(dataset)
    nomes = list(ct.get_feature_names_out())
    col = nomes.index("taxa_chegada__taxa_chegada")

    observado = dataset["taxa_chegada"]
    assert observado.min() > 0.5 or observado.max() < 4.0, (
        "amostra cobriu os limites exatos; o teste perde o sentido"
    )

    # reset_index é necessário: iloc[[0, 0]] repete o RÓTULO 0, e um .loc
    # nesse rótulo escreveria nas duas linhas de uma vez
    extremos = dataset.iloc[[0, 0]].reset_index(drop=True).copy()
    extremos.loc[0, "taxa_chegada"] = 0.5   # low declarado
    extremos.loc[1, "taxa_chegada"] = 4.0   # high declarado
    escalado = ct.transform(extremos)[:, col]

    assert np.allclose(escalado, [0.0, 1.0])


def test_x_categoria_fora_dos_levels_estoura(specs, dataset):
    """handle_unknown='error' é intencional: categoria desconhecida indica
    inconsistência entre DoE e dados, e não pode virar vetor de zeros."""
    ct = build_input_transformer(specs).fit(dataset)
    sujo = dataset.iloc[[0]].copy()
    sujo["regra_despacho"] = pd.Series(["LIFO"], index=sujo.index, dtype="object")

    with pytest.raises(ValueError, match="LIFO|unknown"):
        ct.transform(sujo)


# =====================================================================
# 2. Round-trip de y
# =====================================================================

@pytest.mark.parametrize("method", ["standard", "log_standard", "none"])
def test_round_trip_de_y(specs, dataset, method):
    """inverse_transform_y(transform_y(y)) ≈ y nos três métodos."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", method)])
    train_df, test_df = pipe.split(dataset, test_size=0.25, seed=3)
    pipe.fit(train_df)

    for nome, parte in (("treino", train_df), ("teste", test_df)):
        original = parte["lead_time_mean"].to_numpy(dtype="float64")
        voltou = pipe.inverse_transform_y(pipe.transform_y(parte, "lead_time"), "lead_time")
        assert np.allclose(voltou, original), f"round-trip falhou no {nome} ({method})"


def test_log_standard_aplica_log_antes_de_padronizar(specs, dataset):
    """A ordem interna é log1p -> padronização; a inversa desfaz na ordem
    oposta. Se estivesse trocada, mean_ bateria com a média de y cru."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dataset, test_size=0.25, seed=3)
    pipe.fit(train_df)

    y_train = train_df["lead_time_mean"].to_numpy(dtype="float64")
    alvo = pipe.targets_["lead_time"]

    assert alvo.mean_ == pytest.approx(np.log1p(y_train).mean())
    assert alvo.scale_ == pytest.approx(np.log1p(y_train).std(ddof=0))
    assert alvo.mean_ != pytest.approx(y_train.mean()), (
        "mean_ bateu com a média de y CRU: o log1p não foi aplicado antes"
    )

    # e a transformada é de fato o z-score do log, não o log do z-score
    esperado = (np.log1p(y_train) - alvo.mean_) / alvo.scale_
    assert np.allclose(pipe.transform_y(train_df, "lead_time"), esperado)


def test_none_e_passthrough_real(specs, dataset):
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("throughput", "none")])
    train_df, _ = pipe.split(dataset, test_size=0.25, seed=3)
    pipe.fit(train_df)

    y = train_df["throughput_mean"].to_numpy(dtype="float64")
    assert np.array_equal(pipe.transform_y(train_df, "throughput"), y)


def test_log_standard_recusa_valores_menores_que_menos_um(specs, dataset):
    envenenado = dataset.copy()
    envenenado.loc[envenenado.index[0], "throughput_mean"] = -5.0

    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("throughput", "log_standard")])
    with pytest.raises(ValueError, match="log1p"):
        pipe.fit(envenenado)


def test_method_desconhecido_estoura():
    with pytest.raises(ValueError, match="desconhecido"):
        TargetTransformSpec("lead_time", method="minmax")


# =====================================================================
# 3. Ausência de vazamento
# =====================================================================

def test_fit_de_y_so_ve_o_treino(fitted, dataset):
    """As estatísticas guardadas batem com as de train_df — e NÃO com as
    do dataset completo."""
    pipe, train_df, test_df = fitted

    y_train = train_df["throughput_mean"].to_numpy(dtype="float64")
    y_todos = dataset["throughput_mean"].to_numpy(dtype="float64")
    alvo = pipe.targets_["throughput"]

    assert alvo.mean_ == pytest.approx(y_train.mean())
    assert alvo.scale_ == pytest.approx(y_train.std(ddof=0))
    assert alvo.n_fit_ == len(train_df)

    assert alvo.mean_ != pytest.approx(y_todos.mean()), (
        "mean_ bateu com a média do dataset COMPLETO: o fit viu o teste"
    )

    # o teste é transformado com as estatísticas do treino, não com as dele
    y_test = test_df["throughput_mean"].to_numpy(dtype="float64")
    esperado = (y_test - y_train.mean()) / y_train.std(ddof=0)
    assert np.allclose(pipe.transform_y(test_df, "throughput"), esperado)
    assert not np.allclose(
        pipe.transform_y(test_df, "throughput").mean(), 0.0, atol=1e-12
    ), "y_test centrado em 0: foi padronizado com as estatísticas dele mesmo"


def test_x_do_teste_nao_depende_do_treino(specs, target_specs, dataset):
    """Complemento: X é o mesmo tenha o fit visto o treino ou o dataset
    inteiro — porque X não aprende nada."""
    pipe_a = PreprocessingPipeline(specs, target_specs)
    train_df, test_df = pipe_a.split(dataset, test_size=0.25, seed=7)
    pipe_a.fit(train_df)

    pipe_b = PreprocessingPipeline(specs, target_specs)
    pipe_b.fit(dataset)

    assert np.array_equal(pipe_a.transform_X(test_df), pipe_b.transform_X(test_df))


def test_transform_antes_de_fit_estoura(specs, target_specs, dataset):
    pipe = PreprocessingPipeline(specs, target_specs)
    with pytest.raises(RuntimeError, match="antes de fit"):
        pipe.transform_X(dataset)


def test_split_preserva_sample_weight(fitted):
    """split() devolve DataFrames completos: a coluna de peso continua
    acessível para a Fase 3, mesmo não sendo usada aqui."""
    _, train_df, test_df = fitted
    for kpi in KPIS:
        for parte in (train_df, test_df):
            assert f"{kpi}_sample_weight" in parte.columns
            assert f"{kpi}_sem" in parte.columns
            assert "point_id" in parte.columns


# =====================================================================
# 4. Cobertura de tipos
# =====================================================================

def test_todos_os_tipos_de_spec_produzem_o_numero_certo_de_colunas(fitted, specs):
    """float:1 + integer:1 + categorical(3):3 + ordinal:1 + binary
    ordenado:1 + binary não-ordenado(2):2 = 9 colunas."""
    pipe, train_df, _ = fitted

    X = pipe.transform_X(train_df)
    nomes = pipe.get_feature_names()

    assert X.shape == (len(train_df), 9)
    assert len(nomes) == 9
    assert nomes == [
        "taxa_chegada",
        "num_operadores",
        "regra_despacho_FIFO",
        "regra_despacho_SPT",
        "regra_despacho_EDD",
        "prioridade",
        "turno_extra",
        "modo_setup_0",
        "modo_setup_1",
    ]
    assert np.isfinite(X).all()

    # one-hot é one-hot: exatamente um 1 por linha em cada bloco
    assert np.array_equal(X[:, 2:5].sum(axis=1), np.ones(len(train_df)))
    assert np.array_equal(X[:, 7:9].sum(axis=1), np.ones(len(train_df)))

    # numéricos e ordinal caem em [0,1] pelos limites declarados
    for col in (0, 1, 5, 6):
        assert X[:, col].min() >= 0.0 and X[:, col].max() <= 1.0


def test_parametro_com_underscore_duplo_e_recusado_com_mensagem_util():
    """'__' é o separador do ColumnTransformer. O sklearn recusa o nome,
    mas sem dizer que a origem é o `name` de um ParameterSpec — daí a
    validação própria."""
    specs = [ParameterSpec("taxa__chegada", "float", low=0.0, high=1.0)]
    with pytest.raises(ValueError, match=r"não pode conter '__'"):
        build_input_transformer(specs)


def test_nomes_de_feature_com_prefixo_sobreposto():
    """'turno' é prefixo de 'turno_extra': os rótulos não podem se
    embaralhar na limpeza do prefixo."""
    specs = [
        ParameterSpec("turno", "binary", ordered=True),
        ParameterSpec("turno_extra", "binary", ordered=True),
        ParameterSpec("turno_regra", "categorical", levels=["A", "B"]),
    ]
    df = pd.DataFrame({
        "turno": [0, 1],
        "turno_extra": [1, 0],
        "turno_regra": pd.Categorical(["A", "B"], categories=["A", "B"]),
        "kpi_mean": [3.0, 5.0],
    })
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("kpi")]).fit(df)
    assert pipe.get_feature_names() == [
        "turno", "turno_extra", "turno_regra_A", "turno_regra_B",
    ]


def test_ordinal_preserva_a_ordem_dos_levels(fitted, specs):
    """baixa < media < alta tem de virar 0 < 0.5 < 1 — a monotonicidade é
    justamente o que distingue ordinal de categorical."""
    pipe, train_df, _ = fitted
    col = pipe.get_feature_names().index("prioridade")

    amostra = train_df.iloc[[0, 0, 0]].copy()
    amostra["prioridade"] = pd.Categorical(
        ["baixa", "media", "alta"],
        categories=["baixa", "media", "alta"],
        ordered=True,
    )
    assert np.allclose(pipe.transform_X(amostra)[:, col], [0.0, 0.5, 1.0])


def test_binary_nao_ordenado_chega_como_string(design):
    """Registro do contrato do DoE: ordered=False vira
    CategoricalVariable(['0','1']) e os valores chegam como STRING."""
    assert set(map(type, design["modo_setup"].unique().tolist())) == {str}
    assert design["turno_extra"].dtype.kind == "i"


def test_tipo_de_spec_nao_suportado_estoura():
    spec = ParameterSpec("x", "float", low=0.0, high=1.0)
    spec.type = "quaternion"  # driblando a validação do __post_init__
    with pytest.raises(ValueError, match="não suportado"):
        build_input_transformer([spec])


def test_colunas_de_parametro_ausentes_estouram(fitted):
    pipe, train_df, _ = fitted
    with pytest.raises(ValueError, match="ausentes"):
        pipe.transform_X(train_df.drop(columns=["prioridade"]))


def test_kpi_nao_declarado_estoura(fitted):
    pipe, train_df, _ = fitted
    with pytest.raises(KeyError, match="wip"):
        pipe.transform_y(train_df, "wip")


# =====================================================================
# 5. save / load
# =====================================================================

def test_save_load_reproduz_a_transformacao(fitted, tmp_path):
    pipe, train_df, test_df = fitted
    caminho = tmp_path / "preprocessing.joblib"
    pipe.save(str(caminho))

    recarregado = PreprocessingPipeline.load(str(caminho))

    assert recarregado.get_feature_names() == pipe.get_feature_names()
    assert np.array_equal(recarregado.transform_X(test_df), pipe.transform_X(test_df))
    for kpi in KPIS:
        assert np.array_equal(
            recarregado.transform_y(test_df, kpi), pipe.transform_y(test_df, kpi)
        )
        y_t = pipe.transform_y(test_df, kpi)
        assert np.array_equal(
            recarregado.inverse_transform_y(y_t, kpi),
            pipe.inverse_transform_y(y_t, kpi),
        )


def test_transformador_de_x_e_picklavel(specs):
    """Regressão: escrever os transformadores de X com lambda funciona em
    memória e só quebra em save(), com PicklingError. Este teste falha
    imediatamente se alguém 'simplificar' as classes chamáveis de volta
    para lambdas."""
    import pickle

    ct = build_input_transformer(specs)
    assert pickle.loads(pickle.dumps(ct)) is not None


def test_save_antes_de_fit_estoura(specs, target_specs, tmp_path):
    pipe = PreprocessingPipeline(specs, target_specs)
    with pytest.raises(RuntimeError, match="antes de fit"):
        pipe.save(str(tmp_path / "x.joblib"))


def test_load_de_arquivo_com_outro_objeto_estoura(tmp_path):
    import joblib

    caminho = tmp_path / "outro.joblib"
    joblib.dump({"nao": "e um pipeline"}, caminho)
    with pytest.raises(TypeError, match="não contém"):
        PreprocessingPipeline.load(str(caminho))


# =====================================================================
# 6. Integração — do DoE real até X_train/X_test, y_train/y_test
# =====================================================================

def test_integracao_doe_ate_matrizes(specs, target_specs):
    """Caminho completo usando as funções reais de doe/: generate_design ->
    build_execution_plan -> KPIs sintéticos -> aggregate_replications ->
    este módulo."""
    design = generate_design(specs, n_points=30, seed=11, method="lhs")
    dataset = _simular(specs, design, seed=11)

    # o dataset agregado tem mesmo o formato documentado
    esperadas = {"point_id", *[s.name for s in specs]}
    for kpi in KPIS:
        esperadas |= {f"{kpi}_{s}" for s in ("mean", "std", "n", "sem", "sample_weight")}
    assert set(dataset.columns) == esperadas
    assert len(dataset) == 30

    pipe = PreprocessingPipeline(specs, target_specs)
    train_df, test_df = pipe.split(dataset, test_size=0.2, seed=42)
    pipe.fit(train_df)

    X_train = pipe.transform_X(train_df)
    X_test = pipe.transform_X(test_df)
    y_train = pipe.transform_y(train_df, "lead_time")
    y_test = pipe.transform_y(test_df, "lead_time")

    assert X_train.shape == (24, 9)
    assert X_test.shape == (6, 9)
    assert y_train.shape == (24,)
    assert y_test.shape == (6,)
    assert np.isfinite(X_train).all() and np.isfinite(X_test).all()
    assert np.isfinite(y_train).all() and np.isfinite(y_test).all()

    # nenhum ponto do DoE nos dois lados
    assert set(train_df["point_id"]).isdisjoint(set(test_df["point_id"]))
    assert len(set(train_df["point_id"]) | set(test_df["point_id"])) == 30

    # y de treino padronizado (na escala log): média ~0, desvio ~1
    assert y_train.mean() == pytest.approx(0.0, abs=1e-12)
    assert y_train.std(ddof=0) == pytest.approx(1.0)

    # e a volta para minutos preserva o valor original
    assert np.allclose(
        pipe.inverse_transform_y(y_test, "lead_time"),
        test_df["lead_time_mean"].to_numpy(dtype="float64"),
    )


# =====================================================================
# 7. Peso amostral consistente com a escala do alvo
# =====================================================================

def test_peso_inalterado_quando_a_transformacao_e_afim(specs, dataset):
    """Com "standard" e "none" a transformação do alvo é afim, então ela só
    reescala a variância por uma constante global: os pesos RELATIVOS entre
    pontos não mudam, e não há o que corrigir."""
    for metodo in ("standard", "none"):
        pipe = PreprocessingPipeline(specs, [TargetTransformSpec("throughput", metodo)])
        train_df, _ = pipe.split(dataset, test_size=0.25, seed=5)
        pipe.fit(train_df)

        bruto = train_df["throughput_sample_weight"].to_numpy(dtype="float64")
        convertido = pipe.transform_sample_weight(train_df, "throughput")

        razao = convertido / bruto
        assert np.allclose(razao, razao[0]), (
            f"'{metodo}' alterou os pesos relativos; deveria ser fator global"
        )


def test_log_standard_corrige_o_peso_pela_escala(specs, dataset):
    """Com "log_standard" o peso bruto está na escala errada: ele foi medido
    na escala BRUTA e o modelo treina na escala do log. A correção usa o
    método delta, Var(log1p Y) ≈ Var(Y)/(1+E[Y])², que em peso vira
    multiplicar por (1+média)²."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dataset, test_size=0.25, seed=5)
    pipe.fit(train_df)

    bruto = train_df["lead_time_sample_weight"].to_numpy(dtype="float64")
    media = train_df["lead_time_mean"].to_numpy(dtype="float64")
    esperado = bruto * (1.0 + media) ** 2 * pipe.targets_["lead_time"].scale_ ** 2

    assert np.allclose(pipe.transform_sample_weight(train_df, "lead_time"), esperado)


def test_correcao_do_peso_aumenta_o_tamanho_efetivo_de_amostra(specs, dataset):
    """O efeito prático da correção: o peso bruto concentra o ajuste em
    poucos pontos, e é isso que destrói o metamodelo silenciosamente."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dataset, test_size=0.25, seed=5)
    pipe.fit(train_df)

    def kish(w):
        w = w[np.isfinite(w) & (w > 0)]
        return float(w.sum() ** 2 / np.sum(w**2))

    bruto = kish(train_df["lead_time_sample_weight"].to_numpy(dtype="float64"))
    corrigido = kish(pipe.transform_sample_weight(train_df, "lead_time"))

    assert corrigido > bruto, (
        f"a correção deveria espalhar o peso: {bruto:.2f} -> {corrigido:.2f}"
    )


def test_peso_degenerado_avisa(specs, caplog):
    """Peso muito desigual não levanta exceção e não aparece em lugar nenhum
    até virar métrica ruim na Fase 4. Tem de avisar."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({s.name: _valor_qualquer(s, rng, 20) for s in specs})
    df["kpi_mean"] = rng.uniform(1.0, 100.0, 20)
    # variância proporcional ao quadrado da média -> peso muito desigual
    df["kpi_sample_weight"] = 1.0 / (df["kpi_mean"] ** 2 * 0.03)

    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("kpi", "standard")]).fit(df)
    with caplog.at_level("WARNING"):
        pipe.transform_sample_weight(df, "kpi")

    assert any("Kish" in r.message for r in caplog.records)


def test_peso_ausente_estoura(fitted):
    pipe, train_df, _ = fitted
    with pytest.raises(ValueError, match="sample_weight"):
        pipe.transform_sample_weight(
            train_df.drop(columns=["lead_time_sample_weight"]), "lead_time"
        )


def _valor_qualquer(spec, rng, n):
    """Valores válidos para um spec, só para montar DataFrame de teste."""
    if spec.type == "float":
        return rng.uniform(spec.low, spec.high, n)
    if spec.type == "integer":
        return rng.integers(spec.low, spec.high + 1, n)
    if spec.type in ("categorical", "ordinal"):
        return pd.Categorical(
            rng.choice(spec.levels, n), categories=spec.levels,
            ordered=(spec.type == "ordinal"),
        )
    if spec.ordered:
        return rng.integers(0, 2, n)
    return pd.Categorical(rng.choice(["0", "1"], n), categories=["0", "1"])


# =====================================================================
# 8. Granularidade por replicação e split agrupado
# =====================================================================

@pytest.fixture
def dados_por_replicacao(specs, design):
    """K linhas por ponto, mesmos parâmetros e resultados DIFERENTES — a
    granularidade em que o metamodelo deve ser treinado."""
    plan = build_execution_plan(design, ReplicationPolicy(5, seed_base=0))
    rng = np.random.default_rng(SEED)

    carga = plan["taxa_chegada"].to_numpy(dtype="float64")
    capacidade = plan["num_operadores"].to_numpy(dtype="float64")
    utilizacao = np.clip(carga / (capacidade * 1.4), 0.05, 0.97)

    raw = plan.copy()
    raw["lead_time"] = (
        5.0 * utilizacao / (1.0 - utilizacao)
        * rng.lognormal(0.0, 0.15, size=len(plan))
    )
    return raw


def test_replicas_do_mesmo_ponto_tem_x_igual_e_y_diferente(dados_por_replicacao):
    """A premissa da abordagem: as K réplicas de um ponto são a evidência de
    que aquela entrada não produz UM valor, e sim uma distribuição."""
    ponto = dados_por_replicacao[dados_por_replicacao["point_id"] == 0]

    assert len(ponto) == 5
    assert ponto["taxa_chegada"].nunique() == 1       # X idêntico
    assert ponto["lead_time"].nunique() == 5          # y diferente


def test_split_agrupa_automaticamente_por_ponto(specs, dados_por_replicacao):
    """Com dados por replicação, NENHUM ponto pode aparecer dos dois lados —
    senão o metamodelo é avaliado prevendo um X que já viu no treino."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, test_df = pipe.split(dados_por_replicacao, test_size=0.25, seed=7)

    assert set(train_df["point_id"]).isdisjoint(set(test_df["point_id"]))
    # nenhum ponto foi perdido no caminho
    assert (
        set(train_df["point_id"]) | set(test_df["point_id"])
        == set(dados_por_replicacao["point_id"])
    )
    # e as réplicas de cada ponto ficaram inteiras de um lado só
    for parte in (train_df, test_df):
        assert (parte.groupby("point_id").size() == 5).all()


def test_split_agrupado_evita_metrica_inflada(specs, dados_por_replicacao):
    """O custo concreto de não agrupar: o conjunto de teste passa a conter
    entradas idênticas às do treino, e a métrica deixa de medir
    generalização."""
    from sklearn.model_selection import train_test_split as split_simples

    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    agrupado_treino, agrupado_teste = pipe.split(
        dados_por_replicacao, test_size=0.25, seed=7
    )
    solto_treino, solto_teste = split_simples(
        dados_por_replicacao, test_size=0.25, random_state=7
    )

    vazamento_agrupado = len(
        set(agrupado_treino["point_id"]) & set(agrupado_teste["point_id"])
    )
    vazamento_solto = len(set(solto_treino["point_id"]) & set(solto_teste["point_id"]))

    assert vazamento_agrupado == 0
    assert vazamento_solto > 0, (
        "o split solto deveria contaminar — se não contamina, o teste perdeu o sentido"
    )


def test_alvo_por_replicacao_usa_a_coluna_pura(specs, dados_por_replicacao):
    """Em dado por replicação o alvo é `kpi`, não `kpi_mean` (que nem
    existe)."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dados_por_replicacao, test_size=0.25, seed=7)
    pipe.fit(train_df)

    assert pipe._resolve_target_column(train_df, "lead_time") == "lead_time"
    assert np.allclose(
        pipe.transform_y(train_df, "lead_time"),
        (np.log1p(train_df["lead_time"].to_numpy(dtype="float64"))
         - pipe.targets_["lead_time"].mean_) / pipe.targets_["lead_time"].scale_,
    )


def test_x_identico_para_replicas_do_mesmo_ponto(specs, dados_por_replicacao):
    """As K réplicas têm de produzir exatamente a mesma linha de X — é isso
    que força o metamodelo para a tendência em vez de decorar."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dados_por_replicacao, test_size=0.25, seed=7)
    pipe.fit(train_df)

    primeiro = train_df["point_id"].iloc[0]
    replicas = train_df[train_df["point_id"] == primeiro]
    X = pipe.transform_X(replicas)

    assert len(X) > 1
    assert np.allclose(X, X[0]), "réplicas do mesmo ponto deram X diferente"


def test_peso_amostral_recusa_dado_por_replicacao(specs, dados_por_replicacao):
    """Em dado por replicação não há peso: cada linha é uma observação de
    peso um, e a informação de ruído está nas próprias réplicas."""
    pipe = PreprocessingPipeline(specs, [TargetTransformSpec("lead_time", "log_standard")])
    train_df, _ = pipe.split(dados_por_replicacao, test_size=0.25, seed=7)
    pipe.fit(train_df)

    with pytest.raises(ValueError, match="POR REPLICAÇÃO"):
        pipe.transform_sample_weight(train_df, "lead_time")


def test_dataset_agregado_continua_com_split_simples(fitted):
    """O caminho agregado (1 linha por ponto) não precisa de agrupamento e
    não deve ser penalizado por ele."""
    _, train_df, test_df = fitted

    assert train_df["point_id"].duplicated().sum() == 0
    assert set(train_df["point_id"]).isdisjoint(set(test_df["point_id"]))
