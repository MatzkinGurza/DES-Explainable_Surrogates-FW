"""Fase 4 sobre dados sintéticos de resposta conhecida (tests/sintetico.py)."""

import numpy as np
import openpyxl
import pytest

from sed_meta.doe.replication import compute_noise_ceiling
from sed_meta.evaluation.diagnostics import Limites, diagnose, gatilhos_frame, summarize_cv
from sed_meta.evaluation.metrics import gap_to_noise_ceiling, regression_metrics
from sed_meta.evaluation.record import append_eval_row, eval_record
from sed_meta.evaluation.validation import (
    cross_validate_grouped,
    holdout_evaluation,
    learning_curve_grouped,
    seed_stability,
)
from sed_meta.metamodels.baseline import build_dummy_mean, build_polynomial
from sed_meta.metamodels.tree_family import build_random_forest
from tests.sintetico import KPI, caso


@pytest.fixture(scope="module")
def c():
    return caso()


# ---------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------

def test_regression_metrics_valores_conhecidos():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 6])
    assert m["mse"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse_mae"] == pytest.approx(2.0)      # erro todo concentrado num ponto
    assert m["r2"] == pytest.approx(1 - 4 / 5)


def test_regression_metrics_recusa_nan():
    with pytest.raises(ValueError, match="NaN"):
        regression_metrics([1, np.nan], [1, 2])


def test_gap_to_noise_ceiling():
    g = gap_to_noise_ceiling(0.72, 0.9)
    assert g["gap"] == pytest.approx(0.18) and g["fracao_do_teto"] == pytest.approx(0.8)
    assert np.isnan(gap_to_noise_ceiling(0.5, float("nan"))["gap"])


# ---------------------------------------------------------------------
# validação
# ---------------------------------------------------------------------

def test_cv_agrupada_nao_separa_replicas(c):
    cv = cross_validate_grouped(build_polynomial(2), c.X_train, c.y_train, c.groups_train,
                                n_splits=5, inverse=c.inverse)
    assert set(cv["conjunto"]) == {"treino", "validacao"} and cv["fold"].nunique() == 5
    # cada fold de validação tem pontos inteiros: 3 réplicas por ponto
    va = cv.loc[cv["conjunto"] == "validacao"]
    assert (va["n_linhas"] == 3 * va["n_pontos"]).all()
    assert va["n_pontos"].sum() == len(np.unique(c.groups_train))


def test_holdout_em_escala_original_e_contra_o_teto(c):
    teto = compute_noise_ceiling(c.raw, [KPI])[KPI]
    res = holdout_evaluation(build_polynomial(2), c.X_train, c.y_train, c.X_test, c.y_test,
                             inverse=c.inverse)
    # escala original: RMSE próximo do ruído do simulador (0,5), não de z-scores
    assert 0.3 < res["teste"]["rmse"] < 0.8
    assert res["teste"]["r2"] <= teto + 0.05
    assert res["teste"]["r2"] > 0.8 * teto
    assert np.allclose(res["y_pred_teste"], c.inverse(res["y_pred_teste_t"]))

    piso = holdout_evaluation(build_dummy_mean(), c.X_train, c.y_train, c.X_test, c.y_test,
                              inverse=c.inverse)
    assert piso["teste"]["r2"] < 0.05


def test_learning_curve_aninhada_e_crescente(c):
    lc = learning_curve_grouped(build_polynomial(2), c.X_train, c.y_train, c.groups_train,
                                c.X_test, c.y_test, fractions=(0.25, 0.5, 1.0), inverse=c.inverse)
    assert lc["n_pontos"].is_monotonic_increasing
    assert lc["n_pontos"].iloc[-1] == len(np.unique(c.groups_train))
    assert lc["teste_rmse"].iloc[-1] <= lc["teste_rmse"].iloc[0] + 0.05


def test_seed_stability(c):
    s = seed_stability(lambda seed: build_random_forest(n_estimators=50).set_params(random_state=seed),
                       c.X_train, c.y_train, c.X_test, c.y_test, seeds=(0, 1, 2), inverse=c.inverse)
    assert list(s["seed"]) == [0, 1, 2]
    assert s["r2"].std() < 0.05


# ---------------------------------------------------------------------
# diagnóstico (gatilhos da monografia)
# ---------------------------------------------------------------------

def test_diagnose_modelo_bom_nao_dispara(c):
    teto = compute_noise_ceiling(c.raw, [KPI])[KPI]
    cv = summarize_cv(cross_validate_grouped(build_polynomial(2), c.X_train, c.y_train,
                                             c.groups_train, inverse=c.inverse))
    teste = holdout_evaluation(build_polynomial(2), c.X_train, c.y_train, c.X_test, c.y_test,
                               inverse=c.inverse)["teste"]
    assert diagnose(cv, teste, teto) == []


def test_diagnose_piso_dispara_subajuste(c):
    teto = compute_noise_ceiling(c.raw, [KPI])[KPI]
    cv = summarize_cv(cross_validate_grouped(build_dummy_mean(), c.X_train, c.y_train,
                                             c.groups_train, inverse=c.inverse))
    codigos = [g.codigo for g in diagnose(cv, None, teto)]
    assert "subajuste" in codigos


def test_diagnose_gatilhos_construidos():
    cv = {"n_folds": 5, "treino_r2": 0.98, "validacao_r2": 0.70, "validacao_r2_std": 0.12,
          "validacao_rmse": 1.0, "validacao_mae": 0.5, "gap_r2": 0.28}
    teste = {"r2": 0.95, "rmse": 2.0, "mae": 1.0, "mse": 4.0, "rmse_mae": 2.0}
    achados = diagnose(cv, teste, noise_ceiling=0.80, limites=Limites())
    codigos = {g.codigo for g in achados}
    assert codigos == {"sobreajuste", "instabilidade_cv", "erro_concentrado", "acima_do_teto"}
    tabela = gatilhos_frame(achados)
    assert list(tabela.columns)[:2] == ["codigo", "observacao"] and len(tabela) == 4


# ---------------------------------------------------------------------
# registro na aba eval
# ---------------------------------------------------------------------

def _aba_eval(caminho):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "scenario 1 eval"
    cab = ["Test ID (aammdd-hhmm)", "DOE Size", "# Replications", "Surrogate Model",
           "Model Parametrs (*)", "Promodel simulation time", "Surrogate training time",
           "K-fold # (if used)", "Test Size (%)", "R^2", "RMSE", "MSE", "SHAP", "PDP", "ICE",
           "comments on comparison between promodel and surrogate"]
    for j, nome in enumerate(cab, start=2):
        ws.cell(3, j, value=nome)
    ws["B4"], ws["C4"] = "Example fill ", "261102-2022"
    wb.save(caminho)


def test_append_eval_row_acrescenta_depois_do_exemplo(tmp_path):
    caminho = tmp_path / "rbt.xlsx"
    _aba_eval(caminho)
    teste = {"r2": 0.81234, "rmse": 12.5, "mse": 156.25, "mae": 9.1}
    reg = eval_record(kpi="Tempo_total", surrogate_model="Random forest",
                      model_params={"max_depth": 8}, doe_size=300, n_replications=3,
                      teste=teste, training_time_s=54, k_fold=5, test_size=0.2,
                      test_id="261103-1200")
    assert append_eval_row(caminho, 1, reg) == 5
    assert append_eval_row(caminho, 1, reg) == 6         # nunca sobrescreve

    ws = openpyxl.load_workbook(caminho)["scenario 1 eval"]
    linha = {ws.cell(3, j).value: ws.cell(5, j).value for j in range(2, 18)}
    assert linha["Test ID (aammdd-hhmm)"] == "261103-1200"
    assert linha["R^2"] == pytest.approx(0.8123)
    assert linha["Model Parametrs (*)"] == "max_depth=8"
    assert linha["Surrogate training time"].second == 54
    assert linha["comments on comparison between promodel and surrogate"].startswith(
        "KPI: Tempo_total | MAE: 9.1")
    assert ws["C4"].value == "261102-2022"                # exemplo intacto
