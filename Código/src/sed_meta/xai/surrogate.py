"""
Fase 5 — o metamodelo visto como substituto do simulador.

O metamodelo ajustado prevê no espaço TRANSFORMADO (X escalado pelos
limites do YAML, y padronizado). Para explicar o sistema, as perguntas têm
de ser feitas nas grandezas do problema: "se a Frequência chegada passar de
4 para 8 min, o que acontece com o Tempo_total em segundos?". Esta classe
faz a ponte, passando pelo mesmo PreprocessingPipeline do treino:

    DataFrame com os parâmetros (escala do YAML)
        -> pipe.transform_X -> model.predict -> pipe.inverse_transform_y
        -> previsão na escala original do KPI
"""

import numpy as np
import pandas as pd


class OriginalScaleSurrogate:
    """Chamável: `surrogate(df_parametros) -> np.ndarray` na escala do KPI."""

    def __init__(self, model, pipe, kpi: str):
        self.model = model
        self.pipe = pipe
        self.kpi = kpi
        self.param_names = [s.name for s in pipe.param_specs]

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        y_t = self.model.predict(self.pipe.transform_X(df[self.param_names]))
        return self.pipe.inverse_transform_y(np.asarray(y_t), self.kpi)
