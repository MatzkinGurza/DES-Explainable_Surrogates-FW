"""Testes do fluxo DoE -> plano -> cockpit Excel -> agregação.

Cobre, além do caminho feliz ponta a ponta:
  - preservação de dtype category/ordinal em build_execution_plan(), e a
    forma do plano (ids, unique_id e repetição dos parâmetros);
  - a regra "nada é deletado silenciosamente": linha parcial, pendente e
    inválida continuam no resultado, marcadas, e o descarte só acontece
    no recorte explícito de select_complete_runs().
"""

import logging

import numpy as np
import pandas as pd
import openpyxl
import pytest

from sed_meta.doe.parameter_space import ParameterSpec, generate_design
from sed_meta.doe.replication import (
    ReplicationPolicy,
    aggregate_replications,
    build_execution_plan,
    compute_noise_ceiling,
)
from sed_meta.sim_interface.excel_bridge import (
    DATA_SHEET,
    SCHEMA_SHEET,
    STATUS_COL,
    STATUS_COMPLETA,
    STATUS_INVALIDA,
    STATUS_PARCIAL,
    STATUS_PENDENTE,
    build_cockpit_table,
    extract_results,
    read_cockpit_excel,
    select_complete_runs,
    write_cockpit_excel,
)

KPIS = ["throughput", "lead_time"]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def specs():
    return [
        ParameterSpec("taxa_chegada", "float", low=5.0, high=20.0),
        ParameterSpec("num_maquinas", "integer", low=1, high=5),
        ParameterSpec("regra_despacho", "categorical", levels=["FIFO", "SPT", "EDD"]),
        ParameterSpec("prioridade", "ordinal", levels=["baixa", "media", "alta"]),
        ParameterSpec("turno_extra", "binary"),
    ]


@pytest.fixture
def design(specs):
    return generate_design(specs, n_points=6, seed=42, method="lhs")


@pytest.fixture
def plan(design):
    return build_execution_plan(design, ReplicationPolicy(n_replications=4, id_base=1000))


@pytest.fixture
def cockpit(plan):
    return build_cockpit_table(plan, KPIS)


def _preencher_kpis(xlsx, plan, completas, parciais=(), invalidas=()):
    """Simula o ProModel escrevendo KPIs nas células do .xlsx já salvo.

    completas: índices de linha (0-based no plano) totalmente preenchidos
    parciais : só throughput; lead_time fica em branco
    invalidas: throughput recebe texto não numérico
    """
    wb = openpyxl.load_workbook(xlsx)
    ws = wb[DATA_SHEET]
    col = {c.value: c.column for c in ws[1]}
    rng = np.random.default_rng(7)

    for i in [*completas, *parciais, *invalidas]:
        row = plan.iloc[i]
        base = 40.0 + 3.0 * row["num_maquinas"] - 0.8 * row["taxa_chegada"]
        base += {"FIFO": 0.0, "SPT": 4.0, "EDD": 2.0}[row["regra_despacho"]]
        base += {"baixa": 0.0, "media": 1.5, "alta": 3.0}[row["prioridade"]]
        base += 5.0 * row["turno_extra"]
        throughput = base + rng.normal(0, 1.5)
        lead_time = 120.0 - 1.2 * base + rng.normal(0, 4.0)
        linha = i + 2  # +1 cabeçalho, +1 base-1 do Excel

        if i in invalidas:
            ws.cell(row=linha, column=col["throughput"], value="#N/D")
            ws.cell(row=linha, column=col["lead_time"], value=float(lead_time))
        elif i in parciais:
            ws.cell(row=linha, column=col["throughput"], value=float(throughput))
        else:
            ws.cell(row=linha, column=col["throughput"], value=float(throughput))
            ws.cell(row=linha, column=col["lead_time"], value=float(lead_time))

    wb.save(xlsx)
    wb.close()


# ---------------------------------------------------------------------
# 1. build_execution_plan: forma do plano e dtype preservado
# ---------------------------------------------------------------------

def test_plano_preserva_category_ordenada(design, plan):
    """O fator ordinal tem que chegar ao plano ainda ordenado e com os
    níveis na mesma ordem."""
    assert isinstance(design["prioridade"].dtype, pd.CategoricalDtype)

    dtype = plan["prioridade"].dtype
    assert isinstance(dtype, pd.CategoricalDtype), f"virou {dtype!r}"
    assert dtype.ordered is True
    assert list(dtype.categories) == ["baixa", "media", "alta"]
    # ordem utilizável de verdade, não só declarada
    assert (plan["prioridade"] < "alta").any()


def test_plano_preserva_dtypes_do_design(design, plan):
    for col in design.columns:
        assert plan[col].dtype == design[col].dtype, col


def test_plano_expande_cada_ponto_em_r_execucoes(design):
    """Forma do plano: point_id crescente, replication_id 0..R-1 dentro de
    cada ponto, e os parâmetros repetidos igualmente em todas as réplicas."""
    policy = ReplicationPolicy(n_replications=4, id_base=1000)
    r, n = policy.n_replications, len(design)
    plano = build_execution_plan(design, policy)

    assert list(plano.columns) == ["point_id", "replication_id", "unique_id"] + list(design.columns)
    assert len(plano) == n * r
    assert plano["point_id"].tolist() == [p for p in range(n) for _ in range(r)]
    assert plano["replication_id"].tolist() == [rep for _ in range(n) for rep in range(r)]
    assert plano["unique_id"].tolist() == [
        policy.id_base + p * r + rep for p in range(n) for rep in range(r)
    ]

    for point_id in range(n):
        bloco = plano.loc[plano["point_id"] == point_id, design.columns]
        assert len(bloco) == r
        for col in design.columns:
            esperado = design.iloc[point_id][col]
            assert (bloco[col] == esperado).all(), (point_id, col)


def test_plano_unique_id_deterministico_e_unico(plan):
    assert plan["unique_id"].is_unique
    assert plan["unique_id"].tolist() == list(range(1000, 1000 + len(plan)))


# ---------------------------------------------------------------------
# 2. Round-trip pelo Excel
# ---------------------------------------------------------------------

def test_cockpit_nasce_com_kpis_vazios(plan, cockpit):
    assert list(cockpit.columns) == list(plan.columns) + KPIS
    assert cockpit[KPIS].isna().all().all()


def test_write_gera_abas_e_cabecalho_unico(cockpit, plan, tmp_path):
    xlsx = tmp_path / "cockpit.xlsx"
    write_cockpit_excel(cockpit, str(xlsx), list(plan.columns), KPIS)

    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames[0] == DATA_SHEET
    assert wb[SCHEMA_SHEET].sheet_state == "hidden"
    # cabeçalho na linha 1 e dados já na linha 2 (contrato com o ProModel)
    assert [c.value for c in wb[DATA_SHEET][1]] == list(cockpit.columns)
    assert wb[DATA_SHEET]["A2"].value == 0
    wb.close()


def test_roundtrip_preserva_dtypes(cockpit, plan, tmp_path):
    xlsx = tmp_path / "cockpit.xlsx"
    write_cockpit_excel(cockpit, str(xlsx), list(plan.columns), KPIS)
    lido = read_cockpit_excel(str(xlsx))

    assert list(lido.columns) == list(cockpit.columns)
    for col in cockpit.columns:
        assert lido[col].dtype == cockpit[col].dtype, col

    dtype = lido["prioridade"].dtype
    assert isinstance(dtype, pd.CategoricalDtype)
    assert dtype.ordered is True
    assert list(dtype.categories) == ["baixa", "media", "alta"]


def test_roundtrip_preserva_valores(cockpit, plan, tmp_path):
    xlsx = tmp_path / "cockpit.xlsx"
    write_cockpit_excel(cockpit, str(xlsx), list(plan.columns), KPIS)
    lido = read_cockpit_excel(str(xlsx))

    nao_float = [c for c in plan.columns if cockpit[c].dtype.kind != "f"]
    pd.testing.assert_frame_equal(lido[nao_float], cockpit[nao_float])
    # float: o Excel guarda 15 dígitos significativos, então é isclose, não ==
    assert np.isclose(lido["taxa_chegada"], cockpit["taxa_chegada"]).all()
    assert (lido.groupby("point_id")["taxa_chegada"].nunique() == 1).all()


def test_write_recusa_coluna_nao_classificada(cockpit, plan, tmp_path):
    with pytest.raises(ValueError, match="não classificadas"):
        write_cockpit_excel(cockpit, str(tmp_path / "x.xlsx"),
                            list(plan.columns)[:-1], KPIS)


def test_build_cockpit_recusa_kpi_que_colide_com_parametro(plan):
    with pytest.raises(ValueError, match="colidem"):
        build_cockpit_table(plan, ["taxa_chegada"])


# ---------------------------------------------------------------------
# 3. Regra: nada é deletado silenciosamente
# ---------------------------------------------------------------------

@pytest.fixture
def cockpit_preenchido(cockpit, plan, tmp_path):
    """Cockpit com 18 execuções completas, 1 parcial, 1 inválida e 4 pendentes."""
    xlsx = tmp_path / "cockpit.xlsx"
    write_cockpit_excel(cockpit, str(xlsx), list(plan.columns), KPIS)
    _preencher_kpis(xlsx, plan, completas=range(18), parciais=[18], invalidas=[19])
    return read_cockpit_excel(str(xlsx))


def test_extract_mantem_todas_as_linhas(cockpit_preenchido, plan):
    raw = extract_results(cockpit_preenchido, KPIS)
    assert len(raw) == len(plan) == 24


def test_extract_marca_cada_status(cockpit_preenchido):
    raw = extract_results(cockpit_preenchido, KPIS)
    contagem = raw[STATUS_COL].value_counts().to_dict()
    assert contagem == {
        STATUS_COMPLETA: 18,
        STATUS_PARCIAL: 1,
        STATUS_INVALIDA: 1,
        STATUS_PENDENTE: 4,
    }


def test_linha_parcial_e_mantida_e_marcada(cockpit_preenchido):
    raw = extract_results(cockpit_preenchido, KPIS)
    parcial = raw.iloc[18]
    assert parcial[STATUS_COL] == STATUS_PARCIAL
    assert pd.notna(parcial["throughput"])   # o que foi escrito continua lá
    assert pd.isna(parcial["lead_time"])
    assert parcial["point_id"] == 4          # rastreável até a planilha


def test_linha_pendente_e_mantida_e_marcada(cockpit_preenchido):
    raw = extract_results(cockpit_preenchido, KPIS)
    pendentes = raw[raw[STATUS_COL] == STATUS_PENDENTE]
    assert len(pendentes) == 4
    assert pendentes.index.tolist() == [20, 21, 22, 23]
    assert pendentes[KPIS].isna().all().all()
    # os parâmetros de entrada continuam intactos na linha pendente
    assert pendentes["num_maquinas"].notna().all()


def test_valor_nao_numerico_vira_invalida_sem_sumir(cockpit_preenchido):
    raw = extract_results(cockpit_preenchido, KPIS)
    invalida = raw.iloc[19]
    assert invalida[STATUS_COL] == STATUS_INVALIDA
    assert pd.isna(invalida["throughput"])      # "#N/D" não é número
    assert pd.notna(invalida["lead_time"])      # o KPI válido é preservado


def test_extract_loga_anomalias(cockpit_preenchido, caplog):
    with caplog.at_level(logging.WARNING, logger="sed_meta.sim_interface.excel_bridge"):
        extract_results(cockpit_preenchido, KPIS)
    texto = caplog.text
    assert STATUS_PARCIAL in texto
    assert STATUS_INVALIDA in texto
    assert "#N/D" in texto          # o valor sujo aparece, para achar a célula
    assert "linha_xlsx=21" in texto  # linha da planilha da execução inválida


def test_extract_normaliza_indice_nao_padrao(cockpit_preenchido, caplog):
    """Um DataFrame com índice fora do padrão não pode desalinhar o log
    (que aponta a linha do .xlsx) nem perder linhas."""
    torto = cockpit_preenchido.set_index(
        pd.Index(range(100, 100 + len(cockpit_preenchido)))
    )
    with caplog.at_level(logging.WARNING, logger="sed_meta.sim_interface.excel_bridge"):
        raw = extract_results(torto, KPIS)

    assert len(raw) == 24
    assert raw.index.tolist() == list(range(24))
    assert raw.iloc[18][STATUS_COL] == STATUS_PARCIAL
    assert "linha_xlsx=21" in caplog.text
    assert "#N/D" in caplog.text


def test_select_complete_runs_e_o_unico_descarte(cockpit_preenchido, caplog):
    raw = extract_results(cockpit_preenchido, KPIS)
    with caplog.at_level(logging.WARNING, logger="sed_meta.sim_interface.excel_bridge"):
        completas = select_complete_runs(raw)

    assert len(completas) == 18
    assert (completas[STATUS_COL] == STATUS_COMPLETA).all()
    assert completas[KPIS].notna().all().all()
    assert len(raw) == 24, "o DataFrame de origem não pode ser mutado"
    assert "select_complete_runs" in caplog.text


def test_select_complete_runs_exige_status(cockpit_preenchido):
    with pytest.raises(ValueError, match="status_execucao"):
        select_complete_runs(cockpit_preenchido)


def test_agregacao_avisa_se_receber_incompleta(cockpit_preenchido, specs, caplog):
    """Sem o recorte explícito, a agregação não descarta — ela reclama."""
    raw = extract_results(cockpit_preenchido, KPIS)
    param_cols = [s.name for s in specs]
    with caplog.at_level(logging.WARNING, logger="sed_meta.doe.replication"):
        aggregate_replications(raw, param_cols=param_cols, kpi_cols=KPIS)
    assert "KPI vazio" in caplog.text
    assert "select_complete_runs" in caplog.text


# ---------------------------------------------------------------------
# 4. Ponta a ponta até o dataset de treino
# ---------------------------------------------------------------------

def test_pipeline_completo(cockpit_preenchido, specs):
    raw = extract_results(cockpit_preenchido, KPIS)
    completas = select_complete_runs(raw)
    param_cols = [s.name for s in specs]

    dataset = aggregate_replications(completas, param_cols=param_cols, kpi_cols=KPIS)
    assert len(dataset) == completas["point_id"].nunique()
    for kpi in KPIS:
        assert f"{kpi}_mean" in dataset.columns
        assert f"{kpi}_sem" in dataset.columns
        assert dataset[f"{kpi}_n"].sum() == len(completas)

    ceilings = compute_noise_ceiling(completas, KPIS)
    assert set(ceilings) == set(KPIS)
    for kpi, teto in ceilings.items():
        assert 0.0 <= teto <= 1.0, (kpi, teto)


def test_fallback_sem_aba_de_esquema(cockpit, plan, tmp_path, caplog):
    xlsx = tmp_path / "cockpit.xlsx"
    write_cockpit_excel(cockpit, str(xlsx), list(plan.columns), KPIS)
    _preencher_kpis(xlsx, plan, completas=range(24))

    sem_schema = tmp_path / "sem_schema.xlsx"
    wb = openpyxl.load_workbook(xlsx)
    del wb[SCHEMA_SHEET]
    wb.save(sem_schema)
    wb.close()

    with caplog.at_level(logging.WARNING, logger="sed_meta.sim_interface.excel_bridge"):
        lido = read_cockpit_excel(str(sem_schema))
    assert SCHEMA_SHEET in caplog.text, "cair no best-effort tem que avisar"

    # os inteiros voltam como inteiros mesmo sem esquema...
    assert lido["point_id"].dtype == "int64"
    assert lido["num_maquinas"].dtype == "int64"
    # ...mas a category ordenada não é recuperável — e é por isso que o
    # esquema existe (ver test_roundtrip_preserva_dtypes)
    assert not isinstance(lido["prioridade"].dtype, pd.CategoricalDtype)

    raw = extract_results(lido, KPIS)
    assert len(raw) == 24
    assert (raw[STATUS_COL] == STATUS_COMPLETA).all()
