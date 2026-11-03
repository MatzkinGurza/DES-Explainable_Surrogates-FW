"""
Resultados FALSOS num cockpit — só para exercitar o pipeline de avaliação
antes de o ProModel devolver os resultados de verdade.

Os valores não têm significado físico nenhum e NUNCA vão para
promodel_interface/outputs_of_simulation/ (aquela pasta só recebe o que o
ProModel devolveu). O cockpit falso é gravado em Código/runs/.

Cada saída recebe uma função suave e conhecida das entradas normalizadas
pelos limites do YAML (u_j em [0, 1]):

    y = escala · exp( Σ w_j·u_j + w_int·u_0·u_1 ) · ruído lognormal

com pesos sorteados por saída a partir de `seed`, e ruído diferente em cada
réplica — então as réplicas de um ponto diferem, como num simulador
estocástico.
"""

from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from sed_meta.config import ScenarioConfig
from sed_meta.sim_interface.excel_bridge import PROMODEL_DATA_FIRST_ROW


def fill_with_fake_results(
    cockpit_path: str | Path,
    out_path: str | Path,
    config: ScenarioConfig,
    *,
    seed: int = 0,
    noise_sd: float = 0.10,
) -> dict[str, np.ndarray]:
    """Lê o cockpit gerado, preenche TODAS as saídas e grava em `out_path`
    (o original não é tocado). Devolve os pesos usados por saída
    (w_1..w_d, w_int), para quem quiser conferir a importância verdadeira.
    """
    wb = load_workbook(cockpit_path)
    ws = wb[config.cockpit.sheet_name]
    coluna = {c.value: c.column for c in ws[3] if isinstance(c.value, str)}
    ultima = ws.max_row

    linhas = range(PROMODEL_DATA_FIRST_ROW, ultima + 1)
    U = np.column_stack([
        [(float(ws.cell(r, coluna[f.name]).value) - f.low) / (f.high - f.low) for r in linhas]
        for f in config.factors
    ])

    rng = np.random.default_rng(seed)
    pesos = {}
    for resp in config.responses:
        w = rng.normal(0, 1, U.shape[1]) * (rng.random(U.shape[1]) < 0.6)   # ~40% sem efeito
        w_int = rng.normal(0, 1)
        escala = 10 ** rng.uniform(1, 3)
        y = escala * np.exp(U @ w + w_int * U[:, 0] * U[:, 1]) \
            * np.exp(rng.normal(0, noise_sd, len(U)))
        for r, v in zip(linhas, y):
            ws.cell(r, coluna[resp.name], value=round(float(v), 3))
        pesos[resp.name] = np.append(w, w_int)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return pesos
