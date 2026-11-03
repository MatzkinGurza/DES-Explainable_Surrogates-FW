"""
Configuração de cenário — o YAML em `configs/` validado por pydantic.

O YAML do cenário é a FONTE DE VERDADE da campanha de simulação: dele saem
o espaço de parâmetros do DoE (fatores e limites), as colunas de saída do
cockpit (responses), a política de replicação e o layout da planilha que o
ProModel lê. As abas `scenario N doe` de result-notes/results-by-test.xlsx
são um espelho dele, reescritas por scripts/generate_doe.py — nunca o
contrário.

Duas decisões que valem registro:

1. A ORDEM de `factors` e de `responses` é a ordem das colunas no cockpit.
   A Macro do ProModel lê os cockpits por posição, então a lista não é um
   conjunto: reordenar o YAML reordena a planilha.

2. O arredondamento de float (`decimals`) é aplicado DEPOIS de
   parameter_space.generate_design(), por round_design() aqui — e não dentro
   de parameter_space. O gerador continua amostrando o contínuo (é o que o
   LHS estratifica); a resolução com que o operador/ProModel enxerga o valor
   é uma decisão do cenário, não do método de amostragem. O dtype continua
   float64.

`target` fica nulo nos YAMLs sintéticos: o projeto prevê UM KPI por estudo,
escolhido pelo usuário, e não um modelo que tente prever tudo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sed_meta.doe.parameter_space import ParameterSpec
from sed_meta.sim_interface.excel_bridge import ID_COLS


class _Base(BaseModel):
    # campo digitado errado no YAML tem que falhar, não ser ignorado
    model_config = ConfigDict(extra="forbid")


class ScenarioInfo(_Base):
    name: str
    description: str = ""
    promodel_level: int | None = None


class DoEConfig(_Base):
    method: Literal["lhs", "monte_carlo", "sobol"] = "lhs"
    criterion: str = "ese"
    n_points: int = Field(gt=0)
    seed: int


class ReplicationConfig(_Base):
    n_replications: int = Field(gt=0)


class CockpitLayout(_Base):
    sheet_name: str = "cockpit"
    item_column: str = "Item"
    item_prefix: str = "Teste "


class FactorConfig(_Base):
    name: str
    type: Literal["float", "integer", "categorical", "ordinal", "binary"]
    low: float | None = None
    high: float | None = None
    levels: list | None = None
    decimals: int | None = Field(default=None, ge=0)
    unit: str = ""
    rationale: str = ""

    @model_validator(mode="after")
    def _limites(self):
        if self.type in ("float", "integer"):
            if self.low is None or self.high is None:
                raise ValueError(f"'{self.name}': tipo '{self.type}' exige 'low' e 'high'.")
            if self.low >= self.high:
                raise ValueError(f"'{self.name}': low ({self.low}) precisa ser < high ({self.high}).")
        if self.type in ("categorical", "ordinal") and not self.levels:
            raise ValueError(f"'{self.name}': tipo '{self.type}' exige 'levels'.")
        if self.decimals is not None and self.type != "float":
            raise ValueError(f"'{self.name}': 'decimals' só faz sentido para tipo float.")
        return self


class ResponseConfig(_Base):
    name: str
    unit: str = ""
    description: str = ""


class ScenarioConfig(_Base):
    scenario: ScenarioInfo
    doe: DoEConfig
    replications: ReplicationConfig
    cockpit: CockpitLayout = CockpitLayout()
    factors: list[FactorConfig] = Field(min_length=1)
    responses: list[ResponseConfig] = Field(min_length=1)
    target: str | None = None

    @model_validator(mode="after")
    def _nomes(self):
        nomes = [f.name for f in self.factors] + [r.name for r in self.responses]
        repetidos = sorted({n for n in nomes if nomes.count(n) > 1})
        if repetidos:
            raise ValueError(f"Nomes de coluna repetidos entre factors/responses: {repetidos}")
        reservados = {self.cockpit.item_column, *ID_COLS}
        colisao = sorted(reservados.intersection(nomes))
        if colisao:
            raise ValueError(f"Fator/resposta com nome reservado do layout do cockpit: {colisao}")
        if self.target is not None and self.target not in {r.name for r in self.responses}:
            raise ValueError(f"target '{self.target}' não está em responses.")
        return self

    # ---- atalhos usados pelo script e pelo notebook ------------------

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    @property
    def response_names(self) -> list[str]:
        return [r.name for r in self.responses]

    def to_parameter_specs(self) -> list[ParameterSpec]:
        """Traduz `factors` para a entrada de parameter_space.generate_design(),
        na mesma ordem do YAML."""
        specs = []
        for f in self.factors:
            if f.type == "integer":
                # o floor() de parameter_space conta níveis com limites inteiros
                low, high = int(f.low), int(f.high)
            else:
                low, high = f.low, f.high
            specs.append(ParameterSpec(name=f.name, type=f.type, low=low, high=high,
                                       levels=f.levels))
        return specs


def load_scenario_config(path: str | Path) -> ScenarioConfig:
    """Lê e valida o YAML do cenário. Erro de schema levanta
    pydantic.ValidationError apontando o campo."""
    with open(path, encoding="utf-8") as fh:
        dados = yaml.safe_load(fh)
    return ScenarioConfig.model_validate(dados)


def round_design(design: pd.DataFrame, config: ScenarioConfig) -> pd.DataFrame:
    """Arredonda os fatores float a `decimals` casas (ver item 2 do docstring
    do módulo). Devolve cópia; fatores sem `decimals` passam intactos.

    Se um limite não for representável com aquelas casas (ex.: low=2.05 com
    1 casa), o valor arredondado é preso ao limite arredondado PARA DENTRO,
    para que "limites respeitados" continue verdadeiro depois do arredondamento.
    """
    out = design.copy()
    for f in config.factors:
        if f.type != "float" or f.decimals is None:
            continue
        escala = 10.0 ** f.decimals
        lo = np.ceil(round(f.low * escala, 9)) / escala
        hi = np.floor(round(f.high * escala, 9)) / escala
        out[f.name] = out[f.name].round(f.decimals).clip(lo, hi).astype("float64")
    return out
