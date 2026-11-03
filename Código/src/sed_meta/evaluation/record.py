"""
Fase 4 — registro de um teste na aba `scenario N eval` do results-by-test.

A aba eval é preenchida pelo usuário/pelo código, uma linha por teste, com
o cabeçalho na linha 3 (a partir da coluna B) e uma linha de exemplo na 4.
Este módulo monta o registro e o acrescenta na primeira linha vazia depois
do exemplo — nunca reescreve linhas existentes, e só é chamado quando o
usuário pede (regra do context-for-claude.md: não tocar nas abas eval sem
pedido).

As colunas são achadas pelo COMEÇO do texto do cabeçalho, para tolerar
pequenas variações (ex.: "Model Parametrs (*)"). Cabeçalhos que a aba ainda
não tem — KPI e MAE — vão como prefixo do comentário até a aba ganhar a
coluna; se a coluna existir ("KPI", "MAE"), é preenchida direto.
"""

from datetime import datetime, time, timedelta
from pathlib import Path

# chave do registro -> começo do texto do cabeçalho na linha 3
CABECALHOS = {
    "test_id": "Test ID",
    "doe_size": "DOE Size",
    "n_replications": "# Replications",
    "surrogate_model": "Surrogate Model",
    "model_params": "Model Param",
    "promodel_time": "Promodel simulation time",
    "training_time": "Surrogate training time",
    "k_fold": "K-fold",
    "test_size": "Test Size",
    "r2": "R^2",
    "rmse": "RMSE",
    "mse": "MSE",
    "shap": "SHAP",
    "pdp": "PDP",
    "ice": "ICE",
    "comments": "comments",
    "kpi": "KPI",
    "mae": "MAE",
}
LINHA_CABECALHO = 3
LINHA_EXEMPLO = 4


def _duracao(segundos: float | None):
    """Segundos -> time (como na linha de exemplo da aba); timedelta acima de 24 h."""
    if segundos is None:
        return None
    if segundos < 86400:
        return (datetime.min + timedelta(seconds=round(segundos))).time()
    return timedelta(seconds=round(segundos))


def eval_record(
    *,
    kpi: str,
    surrogate_model: str,
    model_params: dict | str,
    doe_size: int,
    n_replications: int,
    teste: dict,
    training_time_s: float | None = None,
    promodel_time_s: float | None = None,
    k_fold: int | None = None,
    test_size: float | None = None,
    shap: str = "",
    pdp: str = "",
    ice: str = "",
    comments: str = "",
    test_id: str | None = None,
) -> dict:
    """Monta o registro de UM teste (um KPI, um metamodelo).

    `teste` é holdout_evaluation(...)["teste"] (métricas na escala original).
    shap/pdp/ice/comments são texto livre: a interpretação do resultado.
    """
    if isinstance(model_params, dict):
        model_params = ", ".join(f"{k}={v}" for k, v in model_params.items())
    return {
        "test_id": test_id or datetime.now().strftime("%y%m%d-%H%M"),
        "kpi": kpi,
        "doe_size": doe_size,
        "n_replications": n_replications,
        "surrogate_model": surrogate_model,
        "model_params": model_params,
        "promodel_time": _duracao(promodel_time_s),
        "training_time": _duracao(training_time_s),
        "k_fold": k_fold if k_fold else "Not used",
        "test_size": test_size,
        "r2": round(teste["r2"], 4),
        "rmse": round(teste["rmse"], 4),
        "mse": round(teste["mse"], 4),
        "mae": round(teste["mae"], 4),
        "shap": shap,
        "pdp": pdp,
        "ice": ice,
        "comments": comments,
    }


def append_eval_row(xlsx_path: str | Path, scenario_n: int, record: dict) -> int:
    """Acrescenta `record` na aba `scenario {scenario_n} eval`.

    Devolve o número da linha gravada. Campos sem coluna na aba (KPI, MAE)
    entram como prefixo do comentário.
    """
    from openpyxl import load_workbook

    xlsx_path = Path(xlsx_path)
    wb = load_workbook(xlsx_path)
    nome = f"scenario {scenario_n} eval"
    if nome not in wb.sheetnames:
        raise ValueError(f"{xlsx_path.name} não tem a aba '{nome}'.")
    ws = wb[nome]

    colunas = {}
    for cel in ws[LINHA_CABECALHO]:
        if isinstance(cel.value, str):
            for chave, inicio in CABECALHOS.items():
                if chave not in colunas and cel.value.strip().startswith(inicio):
                    colunas[chave] = cel.column
    faltam = [c for c in ("test_id", "r2", "comments") if c not in colunas]
    if faltam:
        raise ValueError(f"Aba '{nome}' sem os cabeçalhos esperados para {faltam} na linha 3.")

    registro = dict(record)
    sem_coluna = [f"{k.upper()}: {registro.pop(k)}" for k in ("kpi", "mae")
                  if k in registro and k not in colunas]
    if sem_coluna:
        registro["comments"] = " | ".join([*sem_coluna, registro.get("comments") or ""]).rstrip(" |")

    linha = LINHA_EXEMPLO + 1
    usadas = list(colunas.values())
    while any(ws.cell(linha, j).value is not None for j in usadas):
        linha += 1
    for chave, valor in registro.items():
        if chave in colunas:
            cel = ws.cell(linha, colunas[chave], value=valor)
            if isinstance(valor, time):
                cel.number_format = "hh:mm:ss"

    wb.save(xlsx_path)
    return linha
