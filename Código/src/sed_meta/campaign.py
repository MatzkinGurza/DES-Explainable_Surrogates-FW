"""
Campanha de DoE — do YAML do cenário ao cockpit entregue ao ProModel.

Encadeia as Fases 1 e 2 na ordem em que scripts/generate_doe.py e
notebooks/generate_doe_walkthrough.ipynb as executam:

    config.load_scenario_config()          YAML validado
    parameter_space.generate_design()      N pontos (contínuo)
    config.round_design()                  resolução de cada fator (decimals)
    replication.build_execution_plan()     N x R linhas
    excel_bridge.build_cockpit_table()     + saídas vazias
    excel_bridge.write_promodel_cockpit()  .xlsx no layout do ProModel
    verify_cockpit()                       relê o arquivo e confere
    write_doe_notes_sheet()                espelho do YAML no results-by-test

Este módulo guarda só o que NÃO cabe nos módulos de fase: o identificador da
campanha, a conferência do arquivo gravado, a metadata e o espelho no
results-by-test.

Onde cada arquivo fica:
    Código/runs/doe_scenario_0N_<DoE-ID>/      cockpit + doe_metadata_<DoE-ID>.json
    promodel_interface/inputs_to_simulation/   SÓ a cópia do cockpit a enviar ao ProModel O script e o notebook chamam as mesmas funções — o notebook
existe para explicar, não para fazer diferente.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sed_meta.config import ScenarioConfig, round_design
from sed_meta.doe.parameter_space import fill_metrics, generate_design
from sed_meta.doe.replication import ReplicationPolicy, build_execution_plan
from sed_meta.sim_interface.excel_bridge import (
    STATUS_COL,
    STATUS_PENDENTE,
    build_cockpit_table,
    extract_results,
    read_promodel_cockpit,
    write_promodel_cockpit,
)

# Código/src/sed_meta/campaign.py -> raiz do repositório
REPO_ROOT = Path(__file__).resolve().parents[3]
INTERFACE_DIR = REPO_ROOT / "promodel_interface"
RUNS_DIR = REPO_ROOT / "Código" / "runs"
RESULTS_NOTES = REPO_ROOT / "result-notes" / "results-by-test.xlsx"


# ---------------------------------------------------------------------
# 1. Identificação e caminhos
# ---------------------------------------------------------------------

def new_doe_id(now: datetime | None = None) -> str:
    """DoE-ID no formato `aammdd-hhmm`, o mesmo do "Test ID" do results-by-test."""
    return (now or datetime.now()).strftime("%y%m%d-%H%M")


def scenario_number(config: ScenarioConfig) -> int:
    """`scenario_02_routing` -> 2. É o N de `scenario N doe` e de `scenario_0N`."""
    m = re.match(r"scenario_(\d+)", config.scenario.name)
    if not m:
        raise ValueError(f"Nome de cenário fora do padrão scenario_NN_*: {config.scenario.name!r}")
    return int(m.group(1))


def interface_input_dir(config: ScenarioConfig) -> Path:
    """Pasta da interface onde ficam os cockpits prontos para o ProModel."""
    return INTERFACE_DIR / "inputs_to_simulation" / f"DoE_cockpits_scenario_{scenario_number(config):02d}"


def run_dir(config: ScenarioConfig, doe_id: str) -> Path:
    """Subpasta de runs/ com o registro completo de uma campanha."""
    return RUNS_DIR / f"doe_scenario_{scenario_number(config):02d}_{doe_id}"


def cockpit_filename(config: ScenarioConfig, doe_id: str) -> str:
    n = scenario_number(config)
    return (f"cockpit_scenario_{n:02d}_{config.doe.method}_"
            f"{config.doe.n_points}x{config.replications.n_replications}_{doe_id}.xlsx")


# ---------------------------------------------------------------------
# 2. Execução da campanha
# ---------------------------------------------------------------------

@dataclass
class CampaignResult:
    doe_id: str
    design: pd.DataFrame            # já arredondado
    plan: pd.DataFrame
    cockpit: pd.DataFrame
    cockpit_path: Path              # em runs/
    interface_path: Path | None     # cópia na promodel_interface (None = não copiado)
    metadata_path: Path
    metadata: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)


def build_design(config: ScenarioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Design cru (saída do gerador) e design arredondado (o que vai ao cockpit)."""
    cru = generate_design(
        config.to_parameter_specs(),
        n_points=config.doe.n_points,
        seed=config.doe.seed,
        method=config.doe.method,
        criterion=config.doe.criterion,
    )
    return cru, round_design(cru, config)


def build_plan_and_cockpit(design: pd.DataFrame, config: ScenarioConfig):
    policy = ReplicationPolicy(
        n_replications=config.replications.n_replications,
    )
    plan = build_execution_plan(design, policy)
    cockpit = build_cockpit_table(plan, config.response_names)
    return plan, cockpit


def write_cockpit(cockpit: pd.DataFrame, path: Path, config: ScenarioConfig) -> None:
    lay = config.cockpit
    write_promodel_cockpit(
        cockpit, str(path), config.factor_names, config.response_names,
        sheet_name=lay.sheet_name, item_column=lay.item_column, item_prefix=lay.item_prefix,
    )


def verify_cockpit(path: Path, config: ScenarioConfig, plan: pd.DataFrame) -> dict:
    """Relê o cockpit gravado e confere o contrato. Levanta AssertionError
    na primeira violação; devolve o resumo das conferências."""
    volta = read_promodel_cockpit(str(path), sheet_name=config.cockpit.sheet_name)
    raw = extract_results(volta, config.response_names)

    n_esperado = config.doe.n_points * config.replications.n_replications
    status = raw[STATUS_COL].value_counts().to_dict()
    assert len(raw) == n_esperado, f"{len(raw)} linhas lidas, esperado {n_esperado}"
    assert status == {STATUS_PENDENTE: n_esperado}, f"status inesperado: {status}"

    n_pontos = raw["point_id"].nunique()
    assert n_pontos == config.doe.n_points, f"{n_pontos} point_ids, esperado {config.doe.n_points}"
    por_ponto = raw.groupby("point_id").size().unique().tolist()
    assert por_ponto == [config.replications.n_replications], f"réplicas por ponto: {por_ponto}"

    variando = int((raw.groupby("point_id")[config.factor_names].nunique() > 1).to_numpy().sum())
    assert variando == 0, f"{variando} fator(es) variando dentro de um mesmo ponto"

    fora = {}
    for f in config.factors:
        if f.type in ("float", "integer"):
            n_fora = int(((raw[f.name] < f.low) | (raw[f.name] > f.high)).sum())
            if n_fora:
                fora[f.name] = n_fora
    assert not fora, f"valores fora dos limites: {fora}"

    for f in config.factors:
        if f.decimals is not None:
            assert np.allclose(raw[f.name], raw[f.name].round(f.decimals)), \
                f"'{f.name}' com mais de {f.decimals} casa(s)"

    for col in ("point_id", "replication_id", "unique_id"):
        assert np.array_equal(raw[col].to_numpy(), plan[col].to_numpy()), f"{col} difere do plano"
    for col in config.factor_names:
        assert np.allclose(raw[col].to_numpy(float), plan[col].to_numpy(float)), \
            f"'{col}' difere do plano"
        assert raw[col].dtype == plan[col].dtype, f"dtype de '{col}': {raw[col].dtype} != {plan[col].dtype}"

    return {
        "linhas": len(raw),
        "status": status,
        "point_ids": int(n_pontos),
        "replicas_por_ponto": config.replications.n_replications,
        "fatores_variando_dentro_do_ponto": variando,
        "valores_fora_dos_limites": 0,
        "dtypes": {c: str(raw[c].dtype) for c in config.factor_names},
    }


def git_info() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "arquivos_modificados": bool(_git("status", "--porcelain")),
    }


def run_campaign(config: ScenarioConfig, out_dir: Path | None = None, *,
                 interface_dir: Path | None = None, config_path: Path | None = None,
                 doe_id: str | None = None) -> CampaignResult:
    """Executa os passos 2–8 do script: design, arredondamento, plano, cockpit,
    conferência e metadata. Não toca no results-by-test.

    O cockpit e o doe_metadata_<DoE-ID>.json vão para `out_dir` (padrão:
    runs/doe_scenario_0N_<DoE-ID>/). Com `interface_dir`, só o cockpit é
    copiado para lá — é a pasta que o ProModel consome.
    """
    inicio = datetime.now()
    doe_id = doe_id or new_doe_id(inicio)
    out_dir = Path(out_dir) if out_dir else run_dir(config, doe_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    nome = cockpit_filename(config, doe_id)
    cockpit_path = out_dir / nome
    meta_path = out_dir / f"doe_metadata_{doe_id}.json"
    interface_path = Path(interface_dir) / nome if interface_dir else None
    existentes = [p for p in (cockpit_path, meta_path, interface_path) if p and p.exists()]
    if existentes:
        raise FileExistsError(f"Já existe(m): {[p.name for p in existentes]}. "
                              "Espere virar o minuto ou passe outro doe_id.")

    t0 = time.perf_counter()
    cru, design = build_design(config)
    t_design = time.perf_counter() - t0

    plan, cockpit = build_plan_and_cockpit(design, config)
    write_cockpit(cockpit, cockpit_path, config)
    checks = verify_cockpit(cockpit_path, config, plan)

    specs = config.to_parameter_specs()
    metadata = {
        "doe_id": doe_id,
        "cenario": config.scenario.name,
        "config_yaml": str(config_path) if config_path else None,
        "config": config.model_dump(mode="json"),
        "git": git_info(),
        "arquivos": {
            "cockpit": nome,
            "copia_interface": str(interface_path) if interface_path else None,
        },
        "execucoes": len(plan),
        "metricas_preenchimento": {
            "design_continuo": fill_metrics(cru, specs),
            "design_arredondado": fill_metrics(design, specs),
        },
        "conferencia_releitura": checks,
        "tempos": {
            "inicio": inicio.isoformat(timespec="seconds"),
            "fim": datetime.now().isoformat(timespec="seconds"),
            "geracao_design_s": round(t_design, 2),
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if interface_path:
        interface_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cockpit_path, interface_path)

    return CampaignResult(doe_id, design, plan, cockpit, cockpit_path, interface_path,
                          meta_path, metadata, checks)


# ---------------------------------------------------------------------
# 3. Espelho do YAML no results-by-test.xlsx
# ---------------------------------------------------------------------

_NOTES_HEADER = ["Coluna", "Papel", "Tipo", "Unidade", "Low", "High", "Casas", "Justificativa / descrição"]


def doe_notes_rows(config: ScenarioConfig) -> list[list]:
    """Tabela de colunas do cockpit (id/entrada/saída), na ordem do arquivo."""
    lay = config.cockpit
    linhas = [
        [lay.item_column, "id", "texto", "", None, None, None,
         f"'{lay.item_prefix}n', n = 1..N = índice da execução (ponteiro da Macro do ProModel)"],
    ]
    for f in config.factors:
        linhas.append([f.name, "entrada", f.type, f.unit, f.low, f.high, f.decimals, f.rationale])
    for r in config.responses:
        linhas.append([r.name, "saída", "float", r.unit, None, None, None, r.description])
    linhas += [
        ["point_id", "id", "integer", "", None, None, None,
         "id do ponto do DoE, constante entre as réplicas do ponto"],
        ["replication_id", "id", "integer", "", None, None, None, "0..R-1 dentro do ponto"],
        ["unique_id", "id", "integer", "", None, None, None, "id único da execução"],
    ]
    return linhas


def write_doe_notes_sheet(config: ScenarioConfig, doe_id: str, cockpit_name: str,
                          xlsx_path: Path = RESULTS_NOTES) -> str:
    """Reescreve SÓ a aba `scenario N doe` a partir do YAML.

    A instrução original em B2 é preservada; tudo da linha 3 para baixo é
    apagado e regravado. As demais abas (inclusive as `eval`) não são tocadas.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    nome_aba = f"scenario {scenario_number(config)} doe"
    wb = load_workbook(xlsx_path)
    if nome_aba not in wb.sheetnames:
        raise ValueError(f"{xlsx_path.name} não tem a aba '{nome_aba}'.")
    ws = wb[nome_aba]

    for rng in list(ws.merged_cells.ranges):
        if rng.min_row >= 3:
            ws.unmerge_cells(str(rng))
    if ws.max_row >= 3:
        ws.delete_rows(3, ws.max_row - 2)

    negrito = Font(bold=True)
    titulo = Font(bold=True, color="FFFFFFFF")
    navy = PatternFill("solid", fgColor="FF002060")
    cor_papel = {"id": "FFBFBFBF", "entrada": "FFC6EFCE", "saída": "FFFFC7CE"}

    linha = 4
    ws.cell(row=linha, column=2, value="Campanha de DoE (espelho do YAML — editar o YAML, não esta aba)").font = negrito
    bloco = [
        ("Cenário", config.scenario.name),
        ("Descrição", config.scenario.description),
        ("Nível ProModel", config.scenario.promodel_level),
        ("Método", config.doe.method),
        ("Critério", config.doe.criterion),
        ("Pontos", config.doe.n_points),
        ("Réplicas", config.replications.n_replications),
        ("Execuções", config.doe.n_points * config.replications.n_replications),
        ("Seed DoE", config.doe.seed),
        ("DoE-ID", doe_id),
        ("Cockpit", cockpit_name),
        ("Target (KPI do estudo)", config.target or "a escolher pelo usuário — um KPI por estudo"),
    ]
    for rotulo, valor in bloco:
        linha += 1
        ws.cell(row=linha, column=2, value=rotulo).font = negrito
        ws.cell(row=linha, column=3, value=valor)

    linha += 2
    for j, nome in enumerate(_NOTES_HEADER, start=2):
        cel = ws.cell(row=linha, column=j, value=nome)
        cel.font, cel.fill = titulo, navy
        cel.alignment = Alignment(horizontal="center")
    for valores in doe_notes_rows(config):
        linha += 1
        for j, v in enumerate(valores, start=2):
            ws.cell(row=linha, column=j, value=v)
        ws.cell(row=linha, column=3).fill = PatternFill("solid", fgColor=cor_papel[valores[1]])

    for letra, largura in zip("BCDEFGHI", (24, 26, 10, 14, 8, 8, 7, 90)):
        ws.column_dimensions[letra].width = largura

    wb.save(xlsx_path)
    return nome_aba
