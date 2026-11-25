"""
Módulos do framework de metamodelagem explicável, na ordem em que
framework.ipynb os usa:

    config.py                     YAML do cenário validado
    doe/parameter_space.py        plano amostral (LHS, Sobol, Monte Carlo)
    doe/replication.py            N pontos -> N x R execuções
    sim_interface/excel_bridge.py cockpit do ProModel: escrita e leitura
    preprocessing/pipeline.py     split agrupado e transformações de X e y
    metamodels/neural_family.py   MLP
    evaluation/metrics.py         R², MAE, RMSE
    evaluation/validation.py      validação cruzada agrupada por ponto
    xai/                          SHAP, PDP, ICE e os gráficos
"""
