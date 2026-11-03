"""Fase 5 — camada de explicabilidade sobre o metamodelo selecionado na Fase 4.

As técnicas da monografia (Tabela "Técnicas de explicabilidade"), mais os
índices de Sobol, que saem baratos quando o simulador é substituído pelo
metamodelo:

    surrogate.py     OriginalScaleSurrogate: parâmetros na escala do YAML ->
                     previsão na escala original do KPI (o que PDP, ICE e
                     Sobol interrogam)
    shap_values.py   SHAP: contribuição de cada variável em cada previsão.
                     TreeSHAP (exato) para as árvores, KernelSHAP (amostral)
                     para os demais
    pdp_ice.py       PDP (efeito médio de uma variável) e ICE (uma curva por
                     ponto; curvas não paralelas = interação que o PDP esconde)
    sobol.py         índices de Sobol de 1ª ordem e total (SALib), sobre os
                     limites declarados do YAML

UM KPI POR VEZ, como na Fase 4: cada explicação é de um modelo de um KPI.

Os resultados alimentam a Validação Teórica (Fase 6): nos cenários
sintéticos, as importâncias devem seguir o que a teoria de filas prevê.
"""
