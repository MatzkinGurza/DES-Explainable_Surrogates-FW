"""
Fase 4 — gatilhos de revisão: dos números para "o que ajustar".

Traduz para código as duas tabelas de gatilhos da monografia ("Gatilhos de
revisão de Metamodelos e Treinamento" e "Gatilhos de revisão da Avaliação
Estatística do Desempenho"). Cada gatilho disparado diz a observação, a
interpretação e o ajuste sugerido no modelo (fluxo 3) e no DoE (fluxo 2):

    sobreajuste          treino bem acima da validação
    subajuste            treino e validação baixos (em relação ao teto de ruído)
    erro_concentrado     RMSE/MAE alto: erro concentrado numa região do espaço
    instabilidade_cv     variância alta da validação entre folds
    instabilidade_seed   variância alta entre sementes
    acima_do_teto        R² de teste acima do teto de ruído: vazamento ou
                         sobreajuste ao ruído (gatilho extra, específico do
                         uso de réplicas)

Os limiares de `Limites` são convenções iniciais, não valores da
literatura: calibrar quando houver resultados reais e registrar a escolha
no context-for-claude.md.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Limites:
    gap_sobreajuste: float = 0.10       # R² treino - R² validação
    fracao_teto_subajuste: float = 0.70  # R² validação < 70% do teto
    r2_subajuste_sem_teto: float = 0.50  # quando o teto não existe (modelo determinístico)
    rmse_mae_concentrado: float = 1.50   # ~1,25 para erro gaussiano
    std_cv: float = 0.05                 # desvio do R² de validação entre folds
    std_seed: float = 0.02               # desvio do R² de teste entre sementes
    tolerancia_teto: float = 0.02        # folga antes de acusar R² acima do teto


@dataclass(frozen=True)
class Gatilho:
    codigo: str
    observacao: str
    interpretacao: str
    ajuste_modelo: str
    ajuste_doe: str
    evidencia: str


def summarize_cv(cv: pd.DataFrame) -> dict[str, float]:
    """Resumo da saída de validation.cross_validate_grouped()."""
    tr = cv.loc[cv["conjunto"] == "treino"]
    va = cv.loc[cv["conjunto"] == "validacao"]
    return {
        "n_folds": int(va["fold"].nunique()),
        "treino_r2": float(tr["r2"].mean()),
        "validacao_r2": float(va["r2"].mean()),
        "validacao_r2_std": float(va["r2"].std(ddof=1)) if len(va) > 1 else float("nan"),
        "validacao_rmse": float(va["rmse"].mean()),
        "validacao_mae": float(va["mae"].mean()),
        "gap_r2": float(tr["r2"].mean() - va["r2"].mean()),
    }


def diagnose(
    cv_resumo: dict | None = None,
    teste: dict | None = None,
    noise_ceiling: float = float("nan"),
    sementes: pd.DataFrame | None = None,
    limites: Limites = Limites(),
) -> list[Gatilho]:
    """Aplica os gatilhos ao que estiver disponível.

    cv_resumo : summarize_cv(cross_validate_grouped(...))
    teste     : holdout_evaluation(...)["teste"]
    noise_ceiling : compute_noise_ceiling(raw, [kpi])[kpi]
    sementes  : seed_stability(...)

    Qualquer argumento pode faltar; o gatilho que depende dele não é
    avaliado. Lista vazia = nenhum gatilho disparado.
    """
    L, achados = limites, []
    tem_teto = bool(np.isfinite(noise_ceiling) and noise_ceiling > 0)

    if cv_resumo is not None:
        gap = cv_resumo["gap_r2"]
        if gap > L.gap_sobreajuste:
            achados.append(Gatilho(
                "sobreajuste",
                "Curva de validação diverge da curva de treinamento",
                "Sobreajuste: não generaliza bem para pontos fora do treino",
                "Simplificar arquitetura ou aumentar regularização",
                "Aumentar o número de pontos amostrais e/ou replicações",
                f"R² treino {cv_resumo['treino_r2']:.3f} vs validação "
                f"{cv_resumo['validacao_r2']:.3f} (gap {gap:.3f} > {L.gap_sobreajuste})",
            ))

        r2_va, r2_tr = cv_resumo["validacao_r2"], cv_resumo["treino_r2"]
        corte = L.fracao_teto_subajuste * noise_ceiling if tem_teto else L.r2_subajuste_sem_teto
        if r2_va < corte and r2_tr < corte + L.gap_sobreajuste:
            referencia = (f"{L.fracao_teto_subajuste:.0%} do teto {noise_ceiling:.3f}"
                          if tem_teto else "limite absoluto (sem teto de ruído)")
            achados.append(Gatilho(
                "subajuste",
                "Curvas de treino e validação convergem para erro elevado",
                "Subajuste: não captura bem a tendência nos dados",
                "Aumentar capacidade do modelo",
                "Revisar a estratégia amostral",
                f"R² treino {r2_tr:.3f}, validação {r2_va:.3f} < {corte:.3f} ({referencia})",
            ))

        std = cv_resumo["validacao_r2_std"]
        if np.isfinite(std) and std > L.std_cv:
            achados.append(Gatilho(
                "instabilidade_cv",
                "Variância elevada entre partições de validação cruzada",
                "Instabilidade durante o treinamento",
                "Revisar hiperparâmetros",
                "Investigar número de replicações",
                f"desvio do R² de validação entre {cv_resumo['n_folds']} folds = {std:.3f} > {L.std_cv}",
            ))

    if teste is not None:
        razao = teste["rmse_mae"]
        if np.isfinite(razao) and razao > L.rmse_mae_concentrado:
            achados.append(Gatilho(
                "erro_concentrado",
                "MAE baixo combinado a RMSE alto",
                "Erro concentrado em região específica do espaço",
                "Revisar hiperparâmetros",
                "Revisar estratégia amostral na região problemática",
                f"RMSE/MAE no teste = {razao:.2f} > {L.rmse_mae_concentrado}",
            ))
        if tem_teto and teste["r2"] > noise_ceiling + L.tolerancia_teto:
            achados.append(Gatilho(
                "acima_do_teto",
                "R² de teste acima do teto de ruído",
                "Vazamento entre treino e teste ou sobreajuste ao ruído",
                "Conferir o split agrupado por point_id e a regularização",
                "Conferir se as réplicas são de fato independentes",
                f"R² teste {teste['r2']:.3f} > teto {noise_ceiling:.3f}",
            ))

    if sementes is not None and len(sementes) > 1:
        std = float(sementes["r2"].std(ddof=1))
        if std > L.std_seed:
            achados.append(Gatilho(
                "instabilidade_seed",
                "Instabilidade entre sementes aleatórias distintas",
                "Treinamento não convergiu de forma confiável",
                "Revisar hiperparâmetros antes de prosseguir",
                "—",
                f"desvio do R² de teste entre {len(sementes)} sementes = {std:.3f} > {L.std_seed}",
            ))

    return achados


def gatilhos_frame(achados: list[Gatilho]) -> pd.DataFrame:
    """Lista de gatilhos como tabela (para notebook e relatório)."""
    colunas = ["codigo", "observacao", "interpretacao", "ajuste_modelo", "ajuste_doe", "evidencia"]
    return pd.DataFrame([vars(g) for g in achados], columns=colunas)
