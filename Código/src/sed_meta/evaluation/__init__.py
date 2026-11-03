"""Fase 4 — avaliação estatística do desempenho dos metamodelos.

Quatro arquivos, na ordem em que a avaliação acontece:

    metrics.py      R², RMSE, MAE, MSE e a distância ao teto de ruído
    validation.py   validação cruzada AGRUPADA por point_id, hold-out na
                    escala original do KPI, curva de aprendizado (erro e
                    tempo de treino x volume de dados) e estabilidade entre
                    sementes
    diagnostics.py  os gatilhos de revisão da monografia (Tabelas
                    "Gatilhos de revisão de Metamodelos" e "da Avaliação
                    Estatística") aplicados aos números acima
    record.py       uma linha da aba `scenario N eval` do results-by-test

UM KPI POR VEZ: toda função recebe o y de UM KPI. O laço sobre KPIs, se
houver, é de quem chama — o projeto estuda um KPI por vez, escolhido pelo
usuário.

ESCALAS: os metamodelos treinam no espaço transformado do preprocessing.
As métricas que vão para a monografia são na escala ORIGINAL do KPI
(segundos, peças): validation.py recebe o `inverse` (em geral
`lambda y: pipe.inverse_transform_y(y, kpi)`) e calcula as duas.
"""
