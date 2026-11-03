"""
generate_doe.py — gera o cockpit de DoE de UM cenário, pronto para o ProModel.

Uso (a partir de Código/):

    ../venv/Scripts/python.exe scripts/generate_doe.py configs/scenario_01_serial.yaml

Passos:
  1. carrega e valida o YAML (sed_meta.config);
  2. gera o design (LHS/ESE por padrão);
  3. arredonda os floats a `decimals` casas;
  4. expande em plano de execução (N pontos x R réplicas);
  5. monta a tabela do cockpit (saídas vazias);
  6. grava o .xlsx no layout dos cockpits de exemplo do ProModel;
  7. relê o arquivo e confere (N x R linhas pendentes, N point_ids, fatores
     constantes dentro do ponto, limites respeitados);
  8. grava doe_metadata_<DoE-ID>.json ao lado do cockpit e copia SÓ o cockpit
     para a promodel_interface;
  9. reescreve a aba `scenario N doe` de result-notes/results-by-test.xlsx.

Saída: Código/runs/doe_scenario_0N_<DoE-ID>/ (cockpit + metadata) e a cópia do
cockpit em promodel_interface/inputs_to_simulation/DoE_cockpits_scenario_0N/.
Notebook-irmão, com o passo a passo explicado:
notebooks/generate_doe_walkthrough.ipynb.
"""

import sys
from pathlib import Path

import click

# layout src/: o script roda sem instalar o pacote
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sed_meta.campaign import (  # noqa: E402
    RESULTS_NOTES,
    interface_input_dir,
    run_campaign,
    write_doe_notes_sheet,
)
from sed_meta.config import load_scenario_config  # noqa: E402


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Pasta do registro (cockpit + metadata). Padrão: runs/doe_scenario_0N_<DoE-ID>/.")
@click.option("--no-interface", is_flag=True, default=False,
              help="Não copiar o cockpit para a promodel_interface.")
@click.option("--no-update-notes", is_flag=True, default=False,
              help="Não reescrever a aba 'scenario N doe' do results-by-test.xlsx.")
@click.option("--notes-path", type=click.Path(dir_okay=False, path_type=Path),
              default=RESULTS_NOTES, show_default=True,
              help="Planilha de registro cuja aba doe é reescrita.")
def main(config_path: Path, out_dir: Path | None, no_interface: bool, no_update_notes: bool,
         notes_path: Path):
    """Gera o cockpit de DoE do cenário descrito em CONFIG_PATH."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_scenario_config(config_path)
    click.echo(f"[1] {config.scenario.name}: {len(config.factors)} fatores, "
               f"{len(config.responses)} saídas")

    interface_dir = None if no_interface else interface_input_dir(config)
    resultado = run_campaign(config, out_dir, interface_dir=interface_dir,
                             config_path=config_path.resolve())
    c = resultado.checks
    click.echo(f"[2-6] design {config.doe.method}/{config.doe.criterion} "
               f"{config.doe.n_points} x {config.replications.n_replications} -> "
               f"{resultado.cockpit_path}")
    click.echo(f"[7] releitura: {c['linhas']} linhas {c['status']}, {c['point_ids']} point_ids, "
               f"{c['fatores_variando_dentro_do_ponto']} fatores variando no ponto, "
               f"{c['valores_fora_dos_limites']} fora dos limites")
    click.echo(f"[8] {resultado.metadata_path}")
    click.echo(f"    cópia para o ProModel: {resultado.interface_path or '(não copiado, --no-interface)'}")

    if no_update_notes:
        click.echo("[9] results-by-test não atualizado (--no-update-notes)")
    else:
        try:
            aba = write_doe_notes_sheet(config, resultado.doe_id, resultado.cockpit_path.name,
                                        notes_path)
        except PermissionError:
            raise click.ClickException(
                f"Não consegui salvar {notes_path} — está aberto no Excel? O cockpit já foi "
                "gerado; feche o arquivo e rode de novo com outro DoE-ID, ou atualize a aba à mão."
            )
        click.echo(f"[9] {notes_path.name}: aba '{aba}' reescrita")
    click.echo(f"DoE-ID: {resultado.doe_id}")


if __name__ == "__main__":
    main()
