"""
O metamodelo visto como substituto do simulador.

O metamodelo prevê no espaço TRANSFORMADO (X escalado pelos limites do
YAML, y padronizado). Esta classe recebe os parâmetros na escala do YAML e
devolve a previsão na escala original do indicador:

    DataFrame de parâmetros -> pipe.transform_X -> model.predict
        -> pipe.inverse_transform_y -> previsão em segundos
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
