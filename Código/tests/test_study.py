"""Estudo de um KPI de ponta a ponta (o que scripts/evaluate_kpi.py faz), em
escala pequena, sobre um cockpit com resultados FALSOS."""

import json

import numpy as np
import openpyxl
import pytest

from sed_meta.campaign import run_campaign
from sed_meta.config import ScenarioConfig
from sed_meta.evaluation.tuning import tune_grouped
from sed_meta.fake_results import fill_with_fake_results
from sed_meta.metamodels.baseline import GRID_POLYNOMIAL_RIDGE, build_dummy_mean, build_polynomial_ridge
from sed_meta.sim_interface.excel_bridge import STATUS_COMPLETA, extract_results, read_promodel_cockpit
from sed_meta.study import MODELOS, eval_record_for, latest_cockpit, run_study, select_best
from tests.sintetico import caso


@pytest.fixture(scope="module")
def config():
    return ScenarioConfig.model_validate({
        "scenario": {"name": "scenario_01_teste", "promodel_level": 1},
        "doe": {"n_points": 30, "seed": 5},
        "replications": {"n_replications": 3},
        "factors": [
            {"name": "Capacidade M1", "type": "integer", "low": 1, "high": 3},
            {"name": "Frequência chegada", "type": "float", "low": 4, "high": 16, "decimals": 1},
            {"name": "Tempo processo 1", "type": "float", "low": 2, "high": 10, "decimals": 1},
        ],
        "responses": [{"name": "Prontas"}, {"name": "Tempo_total"}],
    })


@pytest.fixture(scope="module")
def cockpit_falso(config, tmp_path_factory):
    pasta = tmp_path_factory.mktemp("estudo")
    gerado = run_campaign(config, pasta / "doe", doe_id="261103-0000")
    destino = pasta / "FAKE_cockpit.xlsx"
    pesos = fill_with_fake_results(gerado.cockpit_path, destino, config, seed=1)
    return destino, pesos, gerado.cockpit_path


def test_tune_grouped_grade_vazia_e_sorteio():
    c = caso(n_pontos=40)
    vazio = tune_grouped(build_dummy_mean(), {}, c.X_train, c.y_train, c.groups_train, n_splits=3)
    assert vazio["params"] == {} and vazio["n_candidatos"] == 1
    sorteio = tune_grouped(build_polynomial_ridge(), GRID_POLYNOMIAL_RIDGE, c.X_train, c.y_train,
                           c.groups_train, n_splits=3, n_iter=4)
    assert sorteio["n_candidatos"] == 4 and len(sorteio["cv_results"]) == 4
    assert sorteio["cv_score"] > vazio["cv_score"]


def test_fake_preenche_tudo_e_nao_toca_o_original(config, cockpit_falso):
    destino, pesos, original = cockpit_falso
    raw = extract_results(read_promodel_cockpit(str(destino)), config.response_names)
    assert (raw["status_execucao"] == STATUS_COMPLETA).all() and len(raw) == 90
    # réplicas de um ponto diferem (ruído), como num simulador estocástico
    assert raw.groupby("point_id")["Tempo_total"].nunique().min() > 1
    assert set(pesos) == {"Prontas", "Tempo_total"} and len(pesos["Prontas"]) == 4
    ws = openpyxl.load_workbook(original)["cockpit"]
    assert all(ws.cell(4, j).value is None for j in range(6, 8))      # original continua vazio


def test_run_study_ponta_a_ponta(config, cockpit_falso, tmp_path):
    destino, _, _ = cockpit_falso
    r = run_study(config, "Tempo_total", destino, out_dir=tmp_path / "estudo",
                  modelos=("dummy", "poly_ridge", "random_forest"), n_splits=3, n_iter=2,
                  study_id="261103-0001")
    tabela = r.comparacao["tabela"]
    assert list(tabela["modelo"]) == ["dummy", "poly_ridge", "random_forest"]
    assert r.melhor == select_best(tabela) != "dummy"
    # treino e teste sem ponto em comum
    assert not set(r.prep.train_df["point_id"]) & set(r.prep.test_df["point_id"])

    arquivos = {p.name for p in r.out_dir.iterdir()}
    assert {"resumo.json", "comparacao_modelos.csv", "cv_por_fold.csv", "curva_aprendizado.csv",
            "shap_importancia.csv", "pdp.csv", "modelo.joblib",
            "preprocessing.joblib"} <= arquivos
    resumo = json.loads((r.out_dir / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["kpi"] == "Tempo_total" and resumo["melhor_modelo"] == r.melhor
    assert resumo["status_execucoes"] == {STATUS_COMPLETA: 90}
    assert np.isfinite(resumo["noise_ceiling"])
    assert set(r.xai["shap_importancia"]["parametro"]) == set(config.factor_names)

    reg = eval_record_for(r, config)
    assert reg["kpi"] == "Tempo_total" and reg["surrogate_model"] == r.melhor
    assert reg["shap"].startswith("ordem: ")

    with pytest.raises(FileExistsError):
        run_study(config, "Tempo_total", destino, out_dir=r.out_dir, modelos=("dummy",))


def test_run_study_recusa_kpi_fora_das_responses(config, cockpit_falso, tmp_path):
    with pytest.raises(ValueError, match="não está nas responses"):
        run_study(config, "Inexistente", cockpit_falso[0], out_dir=tmp_path / "x")


def test_latest_cockpit(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_cockpit(tmp_path)
    (tmp_path / "cockpit_a.xlsx").write_bytes(b"")
    assert latest_cockpit(tmp_path).name == "cockpit_a.xlsx"


def test_cardapio_tem_as_tres_familias_da_monografia():
    familias = {f for f, _, _ in MODELOS.values()}
    assert {"árvores", "processo gaussiano", "rede neural", "baseline"} <= familias
