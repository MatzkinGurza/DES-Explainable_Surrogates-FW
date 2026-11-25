# Framework de Metamodelagem Explicável para Análise Paramétrica de Sistemas de Eventos Discretos

Trabalho de Conclusão de Curso realizado pelos alunos Mateus Matzkin Gurza e Vinícius Barile, apresentado ao
Departamento de Engenharia Mecatrônica e de Sistemas Mecânicos da Escola Politécnica da USP. O trabalho propõe
um *framework* que treina metamodelos a partir de simulações no ProModel, para prever as respostas do simulador
em uma fração do tempo da simulação, e aplica técnicas de explicabilidade para quantificar o efeito de cada
parâmetro de entrada.

## Resumo

A Simulação de Eventos Discretos é utilizada na modelagem de sistemas compostos de processos operacionais dinâmicos. A otimização desses sistemas por meio do teste massivo de combinações de parâmetros de entrada possui um custo computacional proibitivo e uma natureza opaca, que oculta a influência exata das variáveis de entrada nas métricas de saída. Para mitigar estes desafios, propõe-se um *framework* integrado ao *software* de simulação ProModel, para treinar e validar Metamodelos (*Surrogate Models*) capazes de aproximar as respostas do simulador com latência inferior à da simulação tradicional. Adicionalmente, uma camada de explicabilidade aplicada ao Metamodelo, quantifica a sensibilidade do sistema com relação a cada variável de entrada. O *framework* busca traçar uma forma de facilitar a tomada de decisões mais rápidas e informadas do que se fossem baseadas na simulação exaustiva. O *framework* foi aplicado a três sistemas de filas sintéticos de complexidade crescente, nos quais o perceptron multicamadas foi o metamodelo de melhor desempenho, com R² médio de validação cruzada entre 0,90 e 0,94, e capacidade de prever 100.000 combinações de parâmetros em segundos, contra dezenas de horas de simulação, além de produzir importâncias coerentes com a teoria de filas.

**Palavras-chave:** Simulação de Eventos Discretos, SED, Metamodelagem, Surrogate Models, Explicabilidade, XAI, ProModel, Projeto de Experimentos, DoE.
