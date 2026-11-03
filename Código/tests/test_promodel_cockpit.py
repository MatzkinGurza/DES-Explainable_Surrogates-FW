"""Round-trip do cockpit no layout do ProModel (write/read_promodel_cockpit).

Cobre: o layout visual padrão (faixas, cabeçalho na linha 3, 'Teste n', cores,
colunas de identificação à direita), o round-trip de valores e dtypes pela aba
oculta de esquema, a leitura pelos ids da própria linha quando as linhas são
reordenadas, o modo best-effort sem a aba oculta, a aceitação de .xlsm e a
compatibilidade da saída com extract_results()/select_complete_runs() sem
adaptação.
"""

import json
import shutil

import numpy as np
import openpyxl
import pytest

from sed_meta.campaign import run_campaign, verify_cockpit, write_doe_notes_sheet
from sed_meta.config import ScenarioConfig
from sed_meta.doe.parameter_space import ParameterSpec, generate_design
from sed_meta.doe.replication import ReplicationPolicy, aggregate_replications, build_execution_plan
from sed_meta.sim_interface.excel_bridge import (
    ID_COLS,
    SCHEMA_SHEET,
    STATUS_COL,
    STATUS_COMPLETA,
    STATUS_PENDENTE,
    build_cockpit_table,
    extract_results,
    read_promodel_cockpit,
    select_complete_runs,
    write_promodel_cockpit,
)

ENTRADAS = ["Capacidade M1", "Frequência chegada", "Tempo processo 1"]
SAIDAS = ["Prontas", "Tempo_total"]
# B = Item, C..E = entradas, F..G = saídas, H..J = ids
COL_SAIDA, COL_ID = 6, 8


@pytest.fixture
def plano():
    specs = [
        ParameterSpec("Capacidade M1", "integer", low=1, high=3),
        ParameterSpec("Frequência chegada", "float", low=2.0, high=8.0),
        ParameterSpec("Tempo processo 1", "float", low=2.0, high=10.0),
    ]
    design = generate_design(specs, n_points=5, seed=7)
    design[["Frequência chegada", "Tempo processo 1"]] = design[
        ["Frequência chegada", "Tempo processo 1"]].round(1)
    return build_execution_plan(design, ReplicationPolicy(n_replications=3, id_base=100))


@pytest.fixture
def arquivo(plano, tmp_path):
    caminho = tmp_path / "cockpit.xlsx"
    write_promodel_cockpit(build_cockpit_table(plano, SAIDAS), str(caminho), ENTRADAS, SAIDAS)
    return caminho


def test_layout_padrao(arquivo, plano):
    wb = openpyxl.load_workbook(arquivo)
    assert wb.sheetnames == ["cockpit", SCHEMA_SHEET]
    assert [wb[s].sheet_state for s in wb.sheetnames] == ["visible", "hidden"]
    ws = wb["cockpit"]

    assert all(ws.cell(1, j).value is None for j in range(1, 11))     # linha 1 vazia
    assert ws["A3"].value is None and ws["A4"].value is None          # coluna A vazia
    assert ws["B2"].value == "Entrada" and ws["B2"].fill.fgColor.rgb == "FF00B050"
    assert ws["F2"].value == "Saída" and ws["F2"].fill.fgColor.rgb == "FFFF0000"
    assert ws["H2"].value == "Identificação" and ws["H2"].fill.fgColor.rgb == "FFBFBFBF"
    assert {str(r) for r in ws.merged_cells.ranges} == {"B2:E2", "F2:G2", "H2:J2"}

    assert [ws.cell(3, j).value for j in range(2, 11)] == ["Item", *ENTRADAS, *SAIDAS, *ID_COLS]
    assert ws["B3"].fill.fgColor.rgb == "FF002060" and ws["B3"].font.b
    assert ws["B3"].font.color.rgb == "FFFFFFFF"
    assert ws["C3"].fill.fill_type is None and ws["C3"].border.left.style == "thin"
    assert ws["H3"].fill.fgColor.rgb == "FFBFBFBF"

    assert ws["B4"].value == "Teste 1" and ws.cell(3 + len(plano), 2).value == f"Teste {len(plano)}"
    assert ws["B4"].fill.fgColor.rgb == "FF002060"
    assert ws["C4"].fill.fgColor.rgb == "FFFFFF00" and ws["C4"].border.left.style == "thin"
    assert ws["F4"].value is None and ws["F4"].fill.fgColor.rgb == "FFFFFF00"  # saída vazia, amarela
    assert ws["H4"].fill.fgColor.rgb == "FFD9D9D9"                             # id, cinza claro
    # ids visíveis: point_id constante no ponto, replication_id 0..R-1, unique_id único
    linhas = [[ws.cell(r, COL_ID + k).value for k in range(3)] for r in range(4, 10)]
    assert linhas == [[0, 0, 100], [0, 1, 101], [0, 2, 102], [1, 0, 103], [1, 1, 104], [1, 2, 105]]


def test_round_trip_vocabulario_interno_e_dtypes(arquivo, plano):
    volta = read_promodel_cockpit(str(arquivo))
    assert list(volta.columns) == [*ID_COLS, *ENTRADAS, *SAIDAS]
    for col in (*ID_COLS, "Capacidade M1"):
        assert volta[col].dtype == "int64"
        assert np.array_equal(volta[col], plano[col])
    for col in ("Frequência chegada", "Tempo processo 1"):
        assert volta[col].dtype == "float64"
        assert np.allclose(volta[col], plano[col])
    assert volta[SAIDAS].isna().all().all()


def test_saida_serve_para_extract_results_sem_adaptacao(arquivo, plano):
    # pendente na ida...
    raw = extract_results(read_promodel_cockpit(str(arquivo)), SAIDAS)
    assert (raw[STATUS_COL] == STATUS_PENDENTE).all() and len(raw) == len(plano)

    # ...completa depois de o "ProModel" preencher as saídas
    wb = openpyxl.load_workbook(arquivo)
    ws = wb["cockpit"]
    rng = np.random.default_rng(0)
    for r in range(4, 4 + len(plano)):
        ws.cell(r, COL_SAIDA, value=int(rng.integers(10, 20)))
        ws.cell(r, COL_SAIDA + 1, value=float(rng.normal(1000, 50)))
    wb.save(arquivo)

    raw = extract_results(read_promodel_cockpit(str(arquivo)), SAIDAS)
    completas = select_complete_runs(raw)
    assert (completas[STATUS_COL] == STATUS_COMPLETA).all() and len(completas) == len(plano)
    agregado = aggregate_replications(completas, ENTRADAS, SAIDAS)
    assert len(agregado) == plano["point_id"].nunique()
    assert (agregado["Tempo_total_n"] == 3).all()


def test_linhas_reordenadas_mantem_os_ids_da_linha(arquivo, plano):
    wb = openpyxl.load_workbook(arquivo)
    ws = wb["cockpit"]
    ultima_linha = 3 + len(plano)
    primeira = [ws.cell(4, j).value for j in range(2, COL_ID + 3)]
    ultima = [ws.cell(ultima_linha, j).value for j in range(2, COL_ID + 3)]
    for j, (a, b) in enumerate(zip(ultima, primeira), start=2):
        ws.cell(4, j, value=a)
        ws.cell(ultima_linha, j, value=b)
    wb.save(arquivo)

    volta = read_promodel_cockpit(str(arquivo))
    for col in ID_COLS:
        assert volta.loc[0, col] == plano[col].iloc[-1]


def test_sem_aba_de_esquema_best_effort(arquivo, plano, caplog):
    wb = openpyxl.load_workbook(arquivo)
    del wb[SCHEMA_SHEET]
    # lixo à direita, separado por coluna sem cabeçalho (como BA:BC do exemplo nível 3)
    wb["cockpit"].cell(4, 13, value="lixo")
    wb.save(arquivo)

    volta = read_promodel_cockpit(str(arquivo))
    assert "sem a aba" in caplog.text
    assert list(volta.columns) == [*ID_COLS, *ENTRADAS, *SAIDAS]
    for col in ID_COLS:
        assert np.array_equal(volta[col], plano[col])
    assert volta["Capacidade M1"].dtype == "int64"
    raw = extract_results(volta, SAIDAS)
    assert (raw[STATUS_COL] == STATUS_PENDENTE).all()


def test_aceita_xlsm(arquivo, tmp_path):
    xlsm = tmp_path / "cockpit.xlsm"
    shutil.copy(arquivo, xlsm)
    assert len(read_promodel_cockpit(str(xlsm))) == 15


def test_le_cockpit_manual_sem_ids(tmp_path):
    """Cockpit montado à mão: coluna A vazia, sem ids e sem aba oculta."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "cockpit"
    ws["B2"], ws["D2"] = "Entrada", "Saída"
    for j, nome in enumerate(["Item", "Capacidade M1", "Prontas"], start=2):
        ws.cell(3, j, value=nome)
    for i in range(3):
        ws.cell(4 + i, 2, value=f"Teste {i + 1}")
        ws.cell(4 + i, 3, value=i + 1)
    caminho = tmp_path / "manual.xlsx"
    wb.save(caminho)

    volta = read_promodel_cockpit(str(caminho))
    assert volta["point_id"].tolist() == [0, 1, 2]
    assert volta["replication_id"].tolist() == [0, 0, 0]
    assert volta["Capacidade M1"].tolist() == [1, 2, 3]


def test_colunas_nao_classificadas_falham(plano, tmp_path):
    cockpit = build_cockpit_table(plano, SAIDAS)
    with pytest.raises(ValueError, match="não classificadas"):
        write_promodel_cockpit(cockpit, str(tmp_path / "x.xlsx"), ENTRADAS[:2], SAIDAS)


# ---------------------------------------------------------------------
# Campanha completa (o que scripts/generate_doe.py faz) em escala pequena
# ---------------------------------------------------------------------

@pytest.fixture
def config():
    return ScenarioConfig.model_validate({
        "scenario": {"name": "scenario_01_teste", "promodel_level": 1},
        "doe": {"n_points": 12, "seed": 3},
        "replications": {"n_replications": 3},
        "factors": [
            {"name": "Capacidade M1", "type": "integer", "low": 1, "high": 3, "rationale": "r"},
            {"name": "Tempo processo 1", "type": "float", "low": 2, "high": 10, "decimals": 1},
        ],
        "responses": [{"name": "Prontas"}, {"name": "Tempo_total"}],
    })


def test_run_campaign_gera_e_confere(config, tmp_path):
    registro, interface = tmp_path / "runs" / "doe", tmp_path / "interface"
    res = run_campaign(config, registro, interface_dir=interface, doe_id="261102-0000")
    nome = "cockpit_scenario_01_lhs_12x3_261102-0000.xlsx"
    assert res.cockpit_path == registro / nome
    assert res.metadata_path == registro / "doe_metadata_261102-0000.json"
    # runs/ guarda cockpit + metadata; a interface recebe SÓ o cockpit
    assert sorted(p.name for p in registro.iterdir()) == sorted([nome, res.metadata_path.name])
    assert [p.name for p in interface.iterdir()] == [nome]
    assert res.interface_path.read_bytes() == res.cockpit_path.read_bytes()
    meta = json.loads(res.metadata_path.read_text(encoding="utf-8"))
    assert meta["arquivos"]["cockpit"] == nome and meta["execucoes"] == 36
    assert res.checks["status"] == {STATUS_PENDENTE: 36}
    assert np.allclose(res.design["Tempo processo 1"], res.design["Tempo processo 1"].round(1))
    with pytest.raises(FileExistsError):
        run_campaign(config, registro, interface_dir=interface, doe_id="261102-0000")


def test_run_campaign_sem_interface(config, tmp_path):
    res = run_campaign(config, tmp_path, doe_id="261102-0003")
    assert res.interface_path is None and res.cockpit_path.exists()


def test_verify_cockpit_detecta_fator_variando_no_ponto(config, tmp_path):
    res = run_campaign(config, tmp_path, doe_id="261102-0001")
    wb = openpyxl.load_workbook(res.cockpit_path)
    ws = wb["cockpit"]
    ws.cell(5, 4, value=ws.cell(4, 4).value + 0.5)     # réplica 2 do ponto 0 diferente
    wb.save(res.cockpit_path)
    with pytest.raises(AssertionError, match="variando"):
        verify_cockpit(res.cockpit_path, config, res.plan)


def test_write_doe_notes_sheet_so_toca_a_aba_doe(config, tmp_path):
    notas = tmp_path / "rbt.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "scenario 1 doe"
    wb["scenario 1 doe"]["B2"] = "instrução original"
    wb["scenario 1 doe"]["B9"] = "conteúdo antigo"
    wb.create_sheet("scenario 1 eval")["B3"] = "Test ID (aammdd-hhmm)"
    wb.save(notas)

    write_doe_notes_sheet(config, "261102-0002", "cockpit.xlsx", notas)
    wb = openpyxl.load_workbook(notas)
    doe = wb["scenario 1 doe"]
    valores = [c for linha in doe.iter_rows(values_only=True) for c in linha if c is not None]
    assert doe["B2"].value == "instrução original"
    assert "conteúdo antigo" not in valores
    assert {"Item", "Capacidade M1", "Tempo processo 1", "Prontas", "Tempo_total",
            *ID_COLS, "261102-0002"} <= set(valores)
    assert wb["scenario 1 eval"]["B3"].value == "Test ID (aammdd-hhmm)"
