"""
evaluate_kpi.py — treina, avalia e explica metamodelos de UM KPI de UM cenário.

Uso (a partir de Código/):

    ../venv/Scripts/python.exe scripts/evaluate_kpi.py configs/scenario_01_serial.yaml --kpi Tempo_total

Passos:
  1. carrega o YAML e o cockpit devolvido pelo ProModel (padrão: o mais recente
     em promodel_interface/outputs_of_simulation/simulation_results_scenario_0N/);
  2. separa as execuções completas e calcula o teto de ruído do KPI;
  3. split AGRUPADO por ponto e preprocessing ajustado só no treino;
  4. por família: busca de hiperparâmetros com CV agrupada, CV na escala
     original, hold-out e gatilhos de revisão;
  5. escolhe o modelo de maior R² de validação cruzada;
  6. curva de aprendizado e estabilidade entre sementes do escolhido;
  7. SHAP e PDP/ICE do escolhido;
  8. grava tudo em Código/runs/eval_scenario_0N_<KPI>_<id>/;
  9. com --register, acrescenta a linha do escolhido na aba `scenario N eval` do
     results-by-test; com --register-all, uma linha por modelo comparado (cada um
     com sua própria XAI, numa versão mais leve), e as colunas extras (KPI, MAE,
     método do DoE, transformação, R² de CV, escolhido, cockpit) à direita.

--fake: antes de começar, preenche uma CÓPIA do cockpit de ENTRADA (o da
promodel_interface/inputs_to_simulation/) com resultados falsos, gravada na
pasta do estudo, e avalia sobre ela. Só para testar o pipeline antes do
ProModel; --fake e --register não andam juntos.

Notebook-irmão, com o passo a passo explicado:
notebooks/evaluate_kpi_walkthrough.ipynb.
"""

import sys
from pathlib import Path

import click

# layout src/: o script roda sem instalar o pacote
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sed_meta.campaign import RESULTS_NOTES, interface_input_dir, scenario_number  # noqa: E402
from sed_meta.config import load_scenario_config  # noqa: E402
from sed_meta.evaluation.record import append_eval_row  # noqa: E402
from sed_meta.fake_results import fill_with_fake_results  # noqa: E402
from sed_meta.study import (  # noqa: E402
    MODELOS,
    MODELOS_PADRAO,
    latest_cockpit,
    results_dir,
    run_study,
    study_dir,
)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--kpi", default=None, help="KPI do estudo (uma das responses). Padrão: `target` do YAML.")
@click.option("--cockpit", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Cockpit devolvido. Padrão: o mais recente da pasta de resultados do cenário.")
@click.option("--models", default=",".join(MODELOS_PADRAO), show_default=True,
              help=f"Modelos a comparar, separados por vírgula. Opções: {', '.join(MODELOS)}.")
@click.option("--target-method", type=click.Choice(["standard", "log_standard", "none"]),
              default="standard", show_default=True, help="Transformação do KPI.")
@click.option("--test-size", default=0.2, show_default=True, help="Fração de PONTOS no hold-out.")
@click.option("--n-splits", default=5, show_default=True, help="Folds da CV agrupada.")
@click.option("--n-iter", default=20, show_default=True,
              help="Combinações sorteadas por família na busca (0 = grade completa).")
@click.option("--seed", default=0, show_default=True)
@click.option("--n-jobs", default=-1, show_default=True,
              help="Processos na busca de hiperparâmetros (-1 = todos os núcleos).")
@click.option("--no-deepen", is_flag=True, default=False,
              help="Pular curva de aprendizado e estabilidade entre sementes (lotes longos).")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Pasta do estudo. Padrão: runs/eval_scenario_0N_<KPI>_<id>/.")
@click.option("--fake", is_flag=True, default=False,
              help="Avaliar sobre uma cópia do cockpit de entrada com resultados falsos.")
@click.option("--register", is_flag=True, default=False,
              help="Acrescentar a linha do escolhido na aba 'scenario N eval' do results-by-test.xlsx.")
@click.option("--register-all", is_flag=True, default=False,
              help="Acrescentar uma linha POR MODELO comparado (com XAI de cada um).")
def main(config_path, kpi, cockpit, models, target_method, test_size, n_splits, n_iter,
         seed, n_jobs, no_deepen, out_dir, fake, register, register_all):
    """Estudo de um KPI do cenário descrito em CONFIG_PATH."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if fake and (register or register_all):
        raise click.UsageError("--fake gera resultados falsos: não registre isso na aba eval.")

    config = load_scenario_config(config_path)
    kpi = kpi or config.target
    if not kpi:
        raise click.UsageError(f"Escolha o KPI com --kpi (o YAML não tem target). "
                               f"Opções: {', '.join(config.response_names)}")
    modelos = [m.strip() for m in models.split(",") if m.strip()]
    desconhecidos = [m for m in modelos if m not in MODELOS]
    if desconhecidos:
        raise click.UsageError(f"Modelos desconhecidos: {desconhecidos}. Opções: {', '.join(MODELOS)}")

    from datetime import datetime
    study_id = datetime.now().strftime("%y%m%d-%H%M")
    out_dir = out_dir or study_dir(config, kpi, study_id)

    if fake:
        origem = cockpit or latest_cockpit(interface_input_dir(config))
        cockpit = out_dir / f"FAKE_{origem.name}"
        fill_with_fake_results(origem, cockpit, config, seed=seed)
        click.echo(f"[1] --fake: resultados falsos em {cockpit}")
    else:
        cockpit = cockpit or latest_cockpit(results_dir(config))
        click.echo(f"[1] cockpit: {cockpit}")

    resultado = run_study(config, kpi, cockpit, out_dir=out_dir, modelos=modelos,
                          target_method=target_method, test_size=test_size, n_splits=n_splits,
                          n_iter=n_iter or None, seed=seed,
                          config_path=config_path.resolve(), study_id=study_id,
                          n_jobs=n_jobs, aprofundar=not no_deepen)
    r = resultado.resumo
    click.echo(f"[2] {kpi}: {r['pontos_completos']} pontos completos, status {r['status_execucoes']}, "
               f"teto de ruído {r['noise_ceiling']:.3f}")
    click.echo(f"[3] split: {r['split']['pontos_treino']}/{r['split']['pontos_teste']} pontos treino/teste")
    tabela = resultado.comparacao["tabela"]
    click.echo("[4] comparação (R² de validação cruzada e de teste, escala original):")
    for _, l in tabela.sort_values("cv_validacao_r2", ascending=False).iterrows():
        click.echo(f"      {l['modelo']:<14} cv {l['cv_validacao_r2']:+.3f}  teste {l['teste_r2']:+.3f}  "
                   f"rmse {l['teste_rmse']:.4g}  gatilhos: {l['gatilhos'] or '—'}")
    click.echo(f"[5] escolhido: {resultado.melhor}  {r['melhor_params']}")
    click.echo("[6] curva de aprendizado e sementes: ver curva_aprendizado.csv / sementes.csv")
    for chave, texto in r["xai_textos"].items():
        click.echo(f"[7] {chave.upper()}: {texto}")
    click.echo(f"[8] estudo gravado em {resultado.out_dir}")

    n = scenario_number(config)
    try:
        if register_all:
            from sed_meta.evaluation.record import ensure_extra_columns
            from sed_meta.study import eval_records_all, explain_all
            textos = explain_all(resultado, config, seed=seed)
            ensure_extra_columns(RESULTS_NOTES, n)
            linhas = [append_eval_row(RESULTS_NOTES, n, reg)
                      for reg in eval_records_all(resultado, config, textos)]
            click.echo(f"[9] {RESULTS_NOTES.name}: linhas {linhas[0]}–{linhas[-1]} da aba 'scenario {n} eval'")
        elif register:
            from sed_meta.study import eval_record_for
            linha = append_eval_row(RESULTS_NOTES, n, eval_record_for(resultado, config))
            click.echo(f"[9] {RESULTS_NOTES.name}: linha {linha} da aba 'scenario {n} eval'")
        else:
            click.echo("[9] aba eval não atualizada (use --register ou --register-all)")
    except PermissionError:
        raise click.ClickException(f"Não consegui salvar {RESULTS_NOTES} — está aberto no Excel? "
                                   "O estudo já foi gravado; feche o arquivo e registre de novo.")
    click.echo(f"Study-ID: {study_id}")


if __name__ == "__main__":
    main()
