"""Testes do schema de cenário (sed_meta/config.py) e dos 3 YAMLs sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sed_meta.config import ScenarioConfig, load_scenario_config, round_design

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SINTETICOS = ["scenario_01_serial", "scenario_02_routing", "scenario_03_multiproduto"]


def _minimo(**extra):
    base = {
        "scenario": {"name": "scenario_09_teste"},
        "doe": {"n_points": 10, "seed": 1},
        "replications": {"n_replications": 2},
        "factors": [{"name": "t", "type": "float", "low": 2.0, "high": 10.0, "decimals": 1},
                    {"name": "c", "type": "integer", "low": 1, "high": 3}],
        "responses": [{"name": "lead_time"}],
    }
    base.update(extra)
    return base


@pytest.mark.parametrize("nome", SINTETICOS)
def test_yamls_sinteticos_validam(nome):
    cfg = load_scenario_config(CONFIGS / f"{nome}.yaml")
    assert cfg.doe.method == "lhs" and cfg.doe.criterion == "ese"
    assert cfg.doe.n_points == 300 and cfg.replications.n_replications == 3
    assert cfg.target is None                      # o KPI do estudo é escolha do usuário
    assert all(f.rationale for f in cfg.factors)   # todo limite justificado
    assert len(cfg.to_parameter_specs()) == len(cfg.factors)


def test_seeds_distintas_entre_cenarios():
    seeds = [load_scenario_config(CONFIGS / f"{n}.yaml").doe.seed for n in SINTETICOS]
    assert len(set(seeds)) == 3


def test_campo_desconhecido_falha():
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(_minimo(replicacoes=3))


def test_limites_invertidos_falham():
    dados = _minimo()
    dados["factors"][0]["low"] = 11.0
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(dados)


def test_decimals_em_inteiro_falha():
    dados = _minimo()
    dados["factors"][1]["decimals"] = 1
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(dados)


def test_nome_reservado_e_target_invalido_falham():
    dados = _minimo()
    dados["factors"][0]["name"] = "Item"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(dados)
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(_minimo(target="nao_existe"))


def test_round_design_arredonda_so_float_e_respeita_limites():
    cfg = ScenarioConfig.model_validate(_minimo())
    design = pd.DataFrame({"t": [2.04, 9.99, 5.55555], "c": np.array([1, 2, 3], dtype="int64")})
    out = round_design(design, cfg)
    assert out["t"].tolist() == [2.0, 10.0, 5.6]
    assert out["t"].dtype == "float64" and out["c"].dtype == "int64"
    assert design["t"].tolist()[0] == 2.04         # não mexe no original
