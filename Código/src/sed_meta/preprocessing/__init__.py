"""Ponte entre a Fase 1/2 (dataset agregado) e a Fase 3 (metamodelos).

Transforma o dataset que sai de doe.replication.aggregate_replications()
nas matrizes X/y que os metamodelos consomem, respeitando a regra que
organiza o módulo inteiro: a transformação das ENTRADAS é determinística
(ancorada nos limites declarados no ParameterSpec, nunca nos observados),
a das SAÍDAS é aprendida e SÓ pode ver o split de treino.

Ver pipeline.py para o detalhamento.
"""

from sed_meta.preprocessing.pipeline import (
    PreprocessingPipeline,
    TargetTransformSpec,
    build_input_transformer,
)

__all__ = [
    "PreprocessingPipeline",
    "TargetTransformSpec",
    "build_input_transformer",
]
