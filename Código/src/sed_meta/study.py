"""
Estudo de UM KPI — do cockpit devolvido pelo ProModel ao metamodelo
avaliado e explicado.

Encadeia as Fases 2 a 5 na ordem em que scripts/evaluate_kpi.py e
notebooks/evaluate_kpi_walkthrough.ipynb as executam:

    load_results()      cockpit -> execuções completas + teto de ruído
    prepare()           split AGRUPADO por ponto, preprocessing ajustado só no treino
    compare_models()    por família: busca agrupada de hiperparâmetros (tuning),
                        CV agrupada na escala original, hold-out, gatilhos
    select_best()       o de maior R² de VALIDAÇÃO cruzada (nunca pelo teste)
    deepen()            curva de aprendizado e estabilidade entre sementes do escolhido
    explain()           SHAP, PDP/ICE e Sobol do escolhido
    write_study()       tudo em Código/runs/eval_scenario_0N_<KPI>_<id>/
    eval_record_for()   a linha da aba eval (gravada só com --register)

UM KPI POR ESTUDO, escolhido pelo usuário — é a regra do projeto. O script
não tem laço sobre KPIs.

Como em campaign.py, este módulo guarda só a orquestração; métricas,
validação, XAI e metamodelos vivem nos módulos de cada fase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from sed_meta.campaign import INTERFACE_DIR, RUNS_DIR, git_info, scenario_number
from sed_meta.config import ScenarioConfig
from sed_meta.doe.replication import compute_noise_ceiling
from sed_meta.evaluation.diagnostics import diagnose, summarize_cv
from sed_meta.evaluation.metrics import gap_to_noise_ceiling
from sed_meta.evaluation.record import eval_record
from sed_meta.evaluation.tuning import tune_grouped
from sed_meta.evaluation.validation import (
    cross_validate_grouped,
    holdout_evaluation,
    learning_curve_grouped,
    seed_stability,
)
from sed_meta.metamodels import baseline, gp_family, neural_family, tree_family
from sed_meta.preprocessing import PreprocessingPipeline, TargetTransformSpec
from sed_meta.sim_interface.excel_bridge import (
    STATUS_COL,
    extract_results,
    read_promodel_cockpit,
    select_complete_runs,
)
from sed_meta.xai.pdp_ice import ice_curves, ice_heterogeneity, pdp_from_ice
from sed_meta.xai.shap_values import compute_shap, importance_by_parameter
from sed_meta.xai.sobol import sobol_indices
from sed_meta.xai.surrogate import OriginalScaleSurrogate

# Cardápio do script: nome na linha de comando -> (família, builder, grade).
# As três famílias da monografia (RF, GP, MLP) + o piso e o polinomial
# regularizado + o boosting, que a monografia deixa como alternativa.
MODELOS = {
    "dummy":        ("baseline", baseline.build_dummy_mean, baseline.GRID_DUMMY_MEAN),
    "poly_ridge":   ("baseline", baseline.build_polynomial_ridge, baseline.GRID_POLYNOMIAL_RIDGE),
    "random_forest": ("árvores", tree_family.build_random_forest, tree_family.GRID_RANDOM_FOREST),
    "xgboost":      ("árvores", tree_family.build_xgboost, tree_family.GRID_XGBOOST),
    "gp":           ("processo gaussiano", gp_family.build_sklearn_gp, gp_family.GRID_SKLEARN_GP),
    "mlp":          ("rede neural", neural_family.build_mlp, neural_family.GRID_MLP),
}
MODELOS_PADRAO = ("dummy", "poly_ridge", "random_forest", "xgboost", "gp", "mlp")


def results_dir(config: ScenarioConfig) -> Path:
    """Pasta da interface onde ficam os cockpits devolvidos pelo ProModel."""
    n = scenario_number(config)
    return INTERFACE_DIR / "outputs_of_simulation" / f"simulation_results_scenario_{n:02d}"


def latest_cockpit(pasta: Path) -> Path:
    """Cockpit mais recente (.xlsx ou .xlsm) de uma pasta."""
    arquivos = sorted([*pasta.glob("cockpit_*.xlsx"), *pasta.glob("cockpit_*.xlsm")],
                      key=lambda p: p.stat().st_mtime)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum cockpit_*.xlsx/.xlsm em {pasta}.")
    return arquivos[-1]


def study_dir(config: ScenarioConfig, kpi: str, study_id: str) -> Path:
    nome_kpi = re.sub(r"\W+", "_", kpi).strip("_")
    return RUNS_DIR / f"eval_scenario_{scenario_number(config):02d}_{nome_kpi}_{study_id}"


# ---------------------------------------------------------------------
# 1. Leitura do cockpit devolvido
# ---------------------------------------------------------------------

def load_results(cockpit_path: Path, config: ScenarioConfig, kpi: str) -> dict:
    """Cockpit -> todas as execuções marcadas, as completas e o teto de ruído.

    Completa = TODAS as saídas do cockpit preenchidas e numéricas (regra de
    extract_results). Nada some em silêncio: as contagens por status vão
    para o resumo do estudo.
    """
    if kpi not in config.response_names:
        raise ValueError(f"KPI '{kpi}' não está nas responses do YAML: {config.response_names}")
    todas = extract_results(read_promodel_cockpit(str(cockpit_path), config.cockpit.sheet_name),
                            config.response_names)
    completas = select_complete_runs(todas)
    if completas.empty:
        raise ValueError(f"{cockpit_path.name}: nenhuma execução completa — o ProModel já rodou?")
    return {
        "todas": todas,
        "completas": completas,
        "status": todas[STATUS_COL].value_counts().to_dict(),
        "noise_ceiling": compute_noise_ceiling(completas, [kpi])[kpi],
    }


# ---------------------------------------------------------------------
# 2. Preparação (split agrupado + preprocessing)
# ---------------------------------------------------------------------

@dataclass
class Preparo:
    kpi: str
    pipe: PreprocessingPipeline
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray

    def inverse(self, y_t):
        return self.pipe.inverse_transform_y(y_t, self.kpi)


def prepare(completas: pd.DataFrame, config: ScenarioConfig, kpi: str, *,
            target_method: str = "standard", test_size: float = 0.2, seed: int = 0) -> Preparo:
    """Split agrupado por point_id (test_size = fração de PONTOS) e
    preprocessing ajustado só no treino. Treina sobre as réplicas."""
    pipe = PreprocessingPipeline(config.to_parameter_specs(), [TargetTransformSpec(kpi, target_method)])
    train_df, test_df = pipe.split(completas, test_size=test_size, seed=seed)
    pipe.fit(train_df)
    return Preparo(kpi, pipe, train_df, test_df,
                   pipe.transform_X(train_df), pipe.transform_y(train_df, kpi),
                   pipe.transform_X(test_df), pipe.transform_y(test_df, kpi),
                   train_df["point_id"].to_numpy())


# ---------------------------------------------------------------------
# 3. Comparação entre famílias
# ---------------------------------------------------------------------

def compare_models(prep: Preparo, modelos=MODELOS_PADRAO, *, noise_ceiling: float = float("nan"),
                   n_splits: int = 5, n_iter: int | None = 20, seed: int = 0) -> dict:
    """Para cada modelo: tuning agrupado -> CV agrupada na escala original ->
    hold-out -> gatilhos. Devolve a tabela comparativa e os detalhes por modelo.

    O hold-out de TODOS os modelos aparece na tabela para a monografia, mas
    a escolha (select_best) usa só a validação cruzada.
    """
    linhas, detalhes = [], {}
    for nome in modelos:
        familia, build, grade = MODELOS[nome]
        tun = tune_grouped(build(), grade, prep.X_train, prep.y_train, prep.groups_train,
                           n_splits=n_splits, n_iter=n_iter, seed=seed)
        melhor = clone(tun["estimador"])
        cv = cross_validate_grouped(melhor, prep.X_train, prep.y_train, prep.groups_train,
                                    n_splits=n_splits, inverse=prep.inverse)
        cv_res = summarize_cv(cv)
        ho = holdout_evaluation(melhor, prep.X_train, prep.y_train, prep.X_test, prep.y_test,
                                inverse=prep.inverse)
        gatilhos = diagnose(cv_res, ho["teste"], noise_ceiling)
        detalhes[nome] = {"tuning": tun, "cv": cv, "cv_resumo": cv_res, "holdout": ho,
                          "gatilhos": gatilhos}
        linhas.append({
            "modelo": nome,
            "familia": familia,
            "params": json.dumps(tun["params"], default=str, ensure_ascii=False),
            "tuning_r2_t": tun["cv_score"],
            "cv_treino_r2": cv_res["treino_r2"],
            "cv_validacao_r2": cv_res["validacao_r2"],
            "cv_validacao_r2_std": cv_res["validacao_r2_std"],
            "teste_r2": ho["teste"]["r2"],
            "teste_rmse": ho["teste"]["rmse"],
            "teste_mae": ho["teste"]["mae"],
            "teste_rmse_mae": ho["teste"]["rmse_mae"],
            "fracao_do_teto": gap_to_noise_ceiling(ho["teste"]["r2"], noise_ceiling)["fracao_do_teto"],
            "fit_time_s": ho["fit_time_s"],
            "tuning_time_s": tun["tempo_s"],
            "n_candidatos": tun["n_candidatos"],
            "gatilhos": ", ".join(g.codigo for g in gatilhos),
        })
    return {"tabela": pd.DataFrame(linhas), "detalhes": detalhes}


def select_best(tabela: pd.DataFrame) -> str:
    """Modelo de maior R² médio na validação cruzada. O dummy só ganha se
    nenhum modelo aprender nada — e isso também é resultado."""
    return str(tabela.sort_values("cv_validacao_r2", ascending=False)["modelo"].iloc[0])


# ---------------------------------------------------------------------
# 4. Aprofundamento do escolhido
# ---------------------------------------------------------------------

def deepen(prep: Preparo, estimador, *, seed: int = 0, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Curva de aprendizado (erro e tempo x volume de pontos) e, se o modelo
    tiver aleatoriedade no treino, estabilidade entre sementes."""
    curva = learning_curve_grouped(clone(estimador), prep.X_train, prep.y_train, prep.groups_train,
                                   prep.X_test, prep.y_test, seed=seed, inverse=prep.inverse)
    sementes = None
    if "random_state" in estimador.get_params():
        sementes = seed_stability(lambda s: clone(estimador).set_params(random_state=s),
                                  prep.X_train, prep.y_train, prep.X_test, prep.y_test,
                                  seeds=seeds, inverse=prep.inverse)
    return {"curva_aprendizado": curva, "sementes": sementes}


# ---------------------------------------------------------------------
# 5. Explicabilidade do escolhido
# ---------------------------------------------------------------------

def explain(prep: Preparo, modelo_ajustado, config: ScenarioConfig, *, seed: int = 0,
            max_pontos_shap: int = 100, n_grid: int = 15, sobol_n: int = 1024) -> dict:
    """SHAP (nos pontos de teste), PDP/ICE (sobre os pontos de treino) e Sobol
    (sobre o espaço do YAML) do metamodelo escolhido."""
    specs = config.to_parameter_specs()
    nomes = prep.pipe.get_feature_names()

    pontos_teste = prep.test_df.drop_duplicates("point_id")
    if len(pontos_teste) > max_pontos_shap:
        pontos_teste = pontos_teste.sample(max_pontos_shap, random_state=seed)
    shap_res = compute_shap(modelo_ajustado, prep.pipe.transform_X(pontos_teste), nomes,
                            X_background=prep.X_train, seed=seed)
    importancia = importance_by_parameter(shap_res, specs)

    surrogate = OriginalScaleSurrogate(modelo_ajustado, prep.pipe, prep.kpi)
    pdps, heterogeneidade = [], {}
    for spec in specs:
        ice = ice_curves(surrogate, prep.train_df, spec, n_grid=n_grid, seed=seed)
        pdps.append(pdp_from_ice(ice).assign(parametro=spec.name))
        heterogeneidade[spec.name] = ice_heterogeneity(ice)
    pdp = pd.concat(pdps, ignore_index=True)[["parametro", "valor", "media", "desvio", "p10", "p90"]]

    return {
        "shap": shap_res,
        "shap_importancia": importancia,
        "pdp": pdp,
        "ice_heterogeneidade": pd.Series(heterogeneidade, name="ice_heterogeneidade"),
        "sobol": sobol_indices(surrogate, specs, n_base=sobol_n, seed=seed),
    }


def xai_texts(xai: dict, top: int = 3) -> dict[str, str]:
    """Resumo em texto para as colunas SHAP/PDP/ICE da aba eval. É um ponto
    de partida: a interpretação de verdade é do usuário."""
    imp = xai["shap_importancia"].head(top)
    shap_txt = "ordem: " + " > ".join(f"{p} ({f:.0%})" for p, f in zip(imp["parametro"], imp["fracao"]))

    efeitos = []
    for p in imp["parametro"]:
        curva = xai["pdp"].loc[xai["pdp"]["parametro"] == p, "media"]
        efeitos.append(f"{p}: {curva.iloc[-1] - curva.iloc[0]:+.3g} de low a high")
    het = xai["ice_heterogeneidade"].dropna().sort_values(ascending=False)
    altas = het[het > 0.05].head(top)
    ice_txt = ("curvas não paralelas (interação) em: " + ", ".join(f"{p} ({v:.2f})" for p, v in altas.items())
               if len(altas) else "curvas aproximadamente paralelas (sem interação forte)")
    return {"shap": shap_txt, "pdp": "; ".join(efeitos), "ice": ice_txt}


# ---------------------------------------------------------------------
# 6. Execução completa e registro
# ---------------------------------------------------------------------

@dataclass
class StudyResult:
    study_id: str
    kpi: str
    cockpit_path: Path
    out_dir: Path
    leitura: dict
    prep: Preparo
    comparacao: dict
    melhor: str
    aprofundamento: dict
    xai: dict
    resumo: dict = field(default_factory=dict)


def run_study(config: ScenarioConfig, kpi: str, cockpit_path: Path, *, out_dir: Path | None = None,
              modelos=MODELOS_PADRAO, target_method: str = "standard", test_size: float = 0.2,
              n_splits: int = 5, n_iter: int | None = 20, seed: int = 0, sobol_n: int = 1024,
              config_path: Path | None = None, study_id: str | None = None) -> StudyResult:
    """Executa os passos 1–6 e grava em `out_dir` (padrão: study_dir())."""
    inicio = datetime.now()
    study_id = study_id or inicio.strftime("%y%m%d-%H%M")
    out_dir = Path(out_dir) if out_dir else study_dir(config, kpi, study_id)
    if (out_dir / "resumo.json").exists():
        raise FileExistsError(f"{out_dir} já tem um estudo (resumo.json).")

    leitura = load_results(Path(cockpit_path), config, kpi)
    prep = prepare(leitura["completas"], config, kpi, target_method=target_method,
                   test_size=test_size, seed=seed)
    comparacao = compare_models(prep, modelos, noise_ceiling=leitura["noise_ceiling"],
                                n_splits=n_splits, n_iter=n_iter, seed=seed)
    melhor = select_best(comparacao["tabela"])
    det = comparacao["detalhes"][melhor]
    aprofundamento = deepen(prep, det["tuning"]["estimador"], seed=seed)
    xai = explain(prep, det["holdout"]["modelo"], config, seed=seed, sobol_n=sobol_n)

    resultado = StudyResult(study_id, kpi, Path(cockpit_path), out_dir, leitura, prep,
                            comparacao, melhor, aprofundamento, xai)
    opcoes = {"modelos": list(modelos), "target_method": target_method, "test_size": test_size,
              "n_splits": n_splits, "n_iter": n_iter, "seed": seed, "sobol_n": sobol_n}
    resultado.resumo = summarize_study(resultado, config, opcoes, config_path=config_path, inicio=inicio)
    write_study(resultado)
    return resultado


def summarize_study(r: StudyResult, config: ScenarioConfig, opcoes: dict, *,
                    config_path: Path | None = None, inicio: datetime | None = None) -> dict:
    """O resumo.json do estudo: de onde vieram os dados, o que foi comparado,
    o escolhido, seus gatilhos e o resumo da explicabilidade."""
    det = r.comparacao["detalhes"][r.melhor]
    return {
        "study_id": r.study_id,
        "cenario": config.scenario.name,
        "kpi": r.kpi,
        "config_yaml": str(config_path) if config_path else None,
        "cockpit": str(r.cockpit_path),
        "git": git_info(),
        "status_execucoes": r.leitura["status"],
        "pontos_completos": int(r.leitura["completas"]["point_id"].nunique()),
        "noise_ceiling": r.leitura["noise_ceiling"],
        "opcoes": opcoes,
        "split": {"pontos_treino": int(r.prep.train_df["point_id"].nunique()),
                  "pontos_teste": int(r.prep.test_df["point_id"].nunique()),
                  "linhas_treino": len(r.prep.train_df), "linhas_teste": len(r.prep.test_df)},
        "melhor_modelo": r.melhor,
        "melhor_params": det["tuning"]["params"],
        "melhor_teste": det["holdout"]["teste"],
        "gatilhos_melhor": [vars(g) for g in det["gatilhos"]],
        "xai_textos": xai_texts(r.xai),
        "tempos": {"inicio": (inicio or datetime.now()).isoformat(timespec="seconds"),
                   "fim": datetime.now().isoformat(timespec="seconds")},
    }


def write_study(r: StudyResult) -> None:
    """Grava o estudo: resumo.json + tabelas .csv + modelo e preprocessing."""
    r.out_dir.mkdir(parents=True, exist_ok=True)
    (r.out_dir / "resumo.json").write_text(
        json.dumps(r.resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    r.comparacao["tabela"].to_csv(r.out_dir / "comparacao_modelos.csv", index=False)
    pd.concat([d["cv"].assign(modelo=m) for m, d in r.comparacao["detalhes"].items()]) \
        .to_csv(r.out_dir / "cv_por_fold.csv", index=False)
    r.aprofundamento["curva_aprendizado"].to_csv(r.out_dir / "curva_aprendizado.csv", index=False)
    if r.aprofundamento["sementes"] is not None:
        r.aprofundamento["sementes"].to_csv(r.out_dir / "sementes.csv", index=False)
    r.xai["shap_importancia"].to_csv(r.out_dir / "shap_importancia.csv", index=False)
    r.xai["shap"].values.to_csv(r.out_dir / "shap_valores.csv", index=False)
    r.xai["pdp"].to_csv(r.out_dir / "pdp.csv", index=False)
    r.xai["ice_heterogeneidade"].to_csv(r.out_dir / "ice_heterogeneidade.csv")
    r.xai["sobol"].to_csv(r.out_dir / "sobol.csv", index=False)
    joblib.dump(r.comparacao["detalhes"][r.melhor]["holdout"]["modelo"], r.out_dir / "modelo.joblib")
    r.prep.pipe.save(str(r.out_dir / "preprocessing.joblib"))


def eval_record_for(r: StudyResult, config: ScenarioConfig, comments: str = "") -> dict:
    """Linha da aba `scenario N eval` para o modelo escolhido."""
    det = r.comparacao["detalhes"][r.melhor]
    textos = r.resumo["xai_textos"]
    return eval_record(
        kpi=r.kpi, surrogate_model=r.melhor, model_params=det["tuning"]["params"],
        doe_size=config.doe.n_points, n_replications=config.replications.n_replications,
        teste=det["holdout"]["teste"], training_time_s=det["holdout"]["fit_time_s"],
        k_fold=r.resumo["opcoes"].get("n_splits"), test_size=r.resumo["opcoes"].get("test_size"),
        shap=textos["shap"], pdp=textos["pdp"], ice=textos["ice"],
        comments=comments or f"teto de ruído {r.leitura['noise_ceiling']:.3f}; estudo em {r.out_dir.name}",
        test_id=r.study_id,
    )
