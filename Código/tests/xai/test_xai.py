"""Fase 5 sobre dados sintéticos de resposta conhecida (tests/sintetico.py):
as explicações têm de recuperar a ordem a > b > regra > c."""

import numpy as np
import pytest

from sed_meta.metamodels.baseline import build_polynomial
from sed_meta.metamodels.tree_family import build_random_forest
from sed_meta.xai.pdp_ice import centered_ice, ice_curves, ice_heterogeneity, parameter_grid, pdp_from_ice
from sed_meta.xai.shap_values import compute_shap, explainer_kind, importance_by_parameter, shap_importance
from sed_meta.xai.sobol import sobol_indices
from sed_meta.xai.surrogate import OriginalScaleSurrogate
from tests.sintetico import KPI, SPECS, caso, resposta

ESPERADO = ["a", "b", "regra", "c"]


@pytest.fixture(scope="module")
def c():
    return caso()


@pytest.fixture(scope="module")
def rf(c):
    return build_random_forest(n_estimators=100).set_params(random_state=0).fit(c.X_train, c.y_train)


@pytest.fixture(scope="module")
def poli(c):
    return build_polynomial(2).fit(c.X_train, c.y_train)


def test_surrogate_responde_na_escala_original(c, poli):
    s = OriginalScaleSurrogate(poli, c.pipe, KPI)
    pontos = c.test_df.drop_duplicates("point_id")
    assert np.sqrt(np.mean((s(pontos) - resposta(pontos)) ** 2)) < 0.5


def test_treeshap_recupera_a_ordem(c, rf):
    assert explainer_kind(rf) == "tree"
    res = compute_shap(rf, c.X_test, c.pipe.get_feature_names())
    assert res.method == "tree" and res.values.shape == (len(c.X_test), len(c.pipe.get_feature_names()))
    # aditividade: base + soma dos SHAP = previsão do modelo
    assert np.allclose(res.base_value + res.values.sum(axis=1), rf.predict(c.X_test), atol=1e-6)
    ordem = importance_by_parameter(res, SPECS)["parametro"].tolist()
    assert ordem[:2] == ["a", "b"] and ordem[-1] == "c"


def test_kernelshap_para_modelo_nao_arvore(c, poli):
    assert explainer_kind(poli) == "kernel"
    with pytest.raises(ValueError, match="X_background"):
        compute_shap(poli, c.X_test[:5], c.pipe.get_feature_names())
    res = compute_shap(poli, c.X_test[:20], c.pipe.get_feature_names(),
                       X_background=c.X_train, n_background=20, nsamples=200)
    imp = shap_importance(res)
    assert imp["fracao"].sum() == pytest.approx(1.0)
    assert importance_by_parameter(res, SPECS)["parametro"].tolist()[:2] == ["a", "b"]


def test_grade_vem_dos_limites_declarados():
    a, _, c_int, regra = SPECS
    assert parameter_grid(a, 5) == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert parameter_grid(c_int) == [1, 2, 3]
    assert parameter_grid(regra) == ["FIFO", "SPT"]


def test_pdp_ice_recupera_os_efeitos(c, poli):
    s = OriginalScaleSurrogate(poli, c.pipe, KPI)
    a, b, c_int, _ = SPECS

    ice_a = ice_curves(s, c.train_df, a, n_grid=11)
    pdp_a = pdp_from_ice(ice_a)
    assert pdp_a["media"].is_monotonic_increasing
    # efeito médio de a de 0 a 1: 8 + 2·E[b] ≈ 9
    assert pdp_a["media"].iloc[-1] - pdp_a["media"].iloc[0] == pytest.approx(9, abs=1.0)
    assert (centered_ice(ice_a).query("valor == 0")["predicao"] == 0).all()

    # c não tem efeito: PDP quase plano
    pdp_c = pdp_from_ice(ice_curves(s, c.train_df, c_int))
    assert pdp_c["media"].max() - pdp_c["media"].min() < 0.5

    # b tem interação com a (termo 2·a·b): as curvas ICE de b abrem em leque
    het_b = ice_heterogeneity(ice_curves(s, c.train_df, b, n_grid=11))
    het_c = ice_heterogeneity(ice_curves(s, c.train_df, c_int))
    assert het_b > 0.05 and het_c < het_b


def test_sobol_recupera_a_ordem(c, poli):
    s = OriginalScaleSurrogate(poli, c.pipe, KPI)
    with pytest.raises(ValueError, match="potência de 2"):
        sobol_indices(s, SPECS, n_base=1000)
    idx = sobol_indices(s, SPECS, n_base=512)
    assert idx["parametro"].tolist()[:2] == ["a", "b"]
    st = idx.set_index("parametro")["ST"]
    assert st["c"] < 0.05 and st["a"] > 0.5
