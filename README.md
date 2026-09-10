# DES-Explainable_Surrogates-FW
Repositório dedicado ao TCC do curso de Engenharia Mecatrônica e Departamento de Engenharia Mecatrônica e Sistemas Mecânicos (PMR) de Mateus Matzkin Gurza e Vinicius Barile


# Framework de Metamodelagem Explicável para Análise Paramétrica de Sistemas de Eventos Discretos
TCC — Escola Politécnica da Universidade de São Paulo (POLI-USP) 



## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Motivação](#motivação)
- [Objetivos](#objetivos)
- [Arquitetura do Framework](#arquitetura-do-framework)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Tecnologias e Dependências](#tecnologias-e-dependências)
- [Instalação e Configuração](#instalação-e-configuração)
- [Como Usar](#como-usar)
- [Módulos](#módulos)
- [Métricas de Validação](#métricas-de-validação)
- [Cronograma](#cronograma)
- [Autores](#autores)
- [Orientador](#orientador)
- [Referências](#referências)

---

## Sobre o Projeto

A **Simulação de Eventos Discretos (SED)** é amplamente utilizada na modelagem de sistemas compostos de processos operacionais dinâmicos. Contudo, a otimização desses sistemas por meio do teste massivo de combinações de parâmetros de entrada possui dois problemas estruturais:

1. **Custo computacional proibitivo** — cada cenário exige uma nova execução completa do simulador.
2. **Opacidade dos resultados** — a simulação não revela a influência exata de cada variável de entrada nas métricas de saída.

Para mitigar esses desafios, este projeto propõe um **framework modular em Python** que consome dados do software de simulação **ProModel**, treina e valida **Metamodelos (Surrogate Models)** capazes de aproximar as respostas do simulador com latência inferior à simulação tradicional e, sobre esses metamodelos, aplica uma **camada de Explicabilidade (XAI)** que quantifica a sensibilidade do sistema em relação a cada variável de entrada.

O framework busca **facilitar a tomada de decisões de forma mais rápida e informada** do que se fossem baseadas na simulação exaustiva.

**Palavras-chave:** Simulação de Eventos Discretos · SED · Metamodelagem · Surrogate Models · Explicabilidade · XAI · ProModel · Projeto de Experimentos · DoE · SHAP · Random Forest · ANN

---

## Motivação

### Interesse Econômico e Social

Sistemas produtivos e logísticos recorrem crescentemente à simulação computacional para embasar decisões estratégicas. O custo de executar milhares de cenários em um simulador de alta fidelidade é, na prática, proibitivo para análises em tempo real ou iterações rápidas de projeto. Um framework que substitua o simulador por um modelo substituto de baixa latência e que ainda explique quantitativamente o comportamento do sistema representa um avanço direto na capacidade analítica de engenheiros e gestores.

### Mérito Técnico-Científico

A literatura demonstra que a sinergia entre Metamodelagem e XAI é tecnicamente viável e apresenta tendência de crescimento. No entanto, sua aplicação ao contexto específico de **SED industrial** — utilizando o ProModel como simulador de referência e integrando DoE, Metamodelagem e XAI em um único pipeline estruturado e reprodutível — permanece pouco explorada. Este trabalho se posiciona exatamente nessa lacuna.

---

## Objetivos

### Produtos Esperados

1. **Pipeline Python** completo, documentado e reutilizável para: geração de dados via DoE no ProModel → treinamento e seleção de metamodelos → aplicação da camada de explicabilidade.
2. **Comparação sistemática** de arquiteturas de metamodelagem (Random Forest, ANN e outros) em termos de acurácia, generalização, tempo de treinamento, latência de inferência e custo computacional.
3. **Conjunto de métricas de fidelidade** que quantificam o grau de aderência do metamodelo ao simulador de referência.
4. **Relatórios de explicabilidade** que correlacionam as importâncias de variáveis extraídas via XAI com a teoria de modelos analíticos de sistemas produtivos e logísticos.
5. **Framework aplicado** em pelo menos dois sistemas: modelos sintéticos de referência e um caso industrial real (dados anonimizados).

### Testes para Validação

| # | Critério | Métrica |
|---|----------|---------|
| 1 | Taxa de acerto das variáveis de saída | R², RMSE, MAE |
| 2 | Taxa de acerto da configuração ótima | Identificação da melhor configuração paramétrica |
| 3 | Tempo de inferência | Razão entre tempo do metamodelo e tempo do simulador |
| 4 | Testes de explicabilidade | Confirmação do efeito de cada variável via replicação no ProModel |

---

## Arquitetura do Framework

O framework integra duas frentes complementares em uma cadeia de processamento modular:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRAMEWORK PIPELINE                           │
│                                                                     │
│  ┌──────────┐   Samples X   ┌────────────┐   Target Y    ┌────────┐ │
│  │   DoE    │ ────────────► │  ProModel  │ ────────────► │  ML    │ │
│  │ Amostrag.│               │ Simulador  │               │ Train  │ │
│  └──────────┘               └────────────┘               └────┬───┘ │
│       ▲                                                       │     │
│  Parâm.                                                       ▼     │
│  Config.                                                 ┌────────┐ │
│                                                          │  XAI   │ │
│                                                          │  Layer │ │
│                                                          └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Módulo 1 — DoE (Design of Experiments):**  
Geração de planos amostrais parametrizados (LHS, Sobol, Full Factorial) para cobertura representativa do espaço paramétrico.

**Módulo 2 — Simulador (ProModel):**  
Execução das simulações nos pontos amostrados. Coleta estruturada dos dados de entrada-saída.

**Módulo 3 — Pré-processamento:**  
Limpeza, normalização e divisão dos dados em conjuntos de treino/validação/teste.

**Módulo 4 — Metamodelagem:**  
Treinamento e comparação de arquiteturas (Random Forest, ANN, Gradient Boosting, Kriging/GP). Seleção via métricas de fidelidade.

**Módulo 5 — Explicabilidade (XAI):**  
Aplicação de SHAP *(Shapley Additive Explanations)*, PDP *(Partial Dependence Plots)* e ICE *(Individual Conditional Expectations)* sobre o metamodelo selecionado.

---

## Estrutura do Repositório

```
📦 DES-Explainable_Surrogates-FW/
│
├── 📂 data/                        # Dados gerados e utilizados
│   ├── raw/                        # Saídas brutas do ProModel
│   ├── processed/                  # Dados pré-processados
│   └── synthetic/                  # Casos sintéticos de referência
│
├── 📂 doe/                         # Módulo de Projeto de Experimentos
│   ├── sampling.py                 # LHS, Sobol, Full Factorial
│   ├── space_definition.py         # Definição do espaço paramétrico
│   └── README.md
│
├── 📂 simulation/                  # Interface com o ProModel
│   ├── promodel_interface.py       # Automação de execução
│   ├── data_collector.py           # Coleta e parsing de resultados
│   └── README.md
│
├── 📂 preprocessing/               # Pré-processamento dos dados
│   ├── cleaner.py
│   ├── normalizer.py
│   └── splitter.py
│
├── 📂 metamodeling/                # Módulo de Metamodelagem
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── neural_network.py
│   │   ├── gradient_boosting.py
│   │   └── kriging.py
│   ├── trainer.py                  # Pipeline de treinamento
│   ├── validator.py                # Métricas de fidelidade
│   └── README.md
│
├── 📂 xai/                         # Camada de Explicabilidade
│   ├── shap_analysis.py            # SHAP global e local
│   ├── pdp_analysis.py             # Partial Dependence Plots
│   ├── ice_analysis.py             # Individual Conditional Expectations
│   └── README.md
│
├── 📂 experiments/                 # Experimentos executados
│   ├── synthetic_case_1/
│   ├── synthetic_case_2/
│   └── industrial_case/            # Dados anonimizados
│
├── 📂 results/                     # Resultados, métricas e visualizações
│   ├── metrics/
│   ├── plots/
│   └── logs/                       # Logs de rastreabilidade por execução
│
├── 📂 monografia/                  # Documentação acadêmica (LaTeX)
│   ├── TCC_Document.tex
│   └── refs.bib
│
├── 📂 notebooks/                   # Jupyter Notebooks exploratórios
│   ├── 01_doe_exploration.ipynb
│   ├── 02_metamodel_benchmark.ipynb
│   └── 03_xai_analysis.ipynb
│
├── 📂 tests/                       # Testes unitários e de integração
│
├── config.yaml                     # Configuração central do framework
├── main.py                         # Ponto de entrada principal
├── requirements.txt
└── README.md
```

---

## Tecnologias e Dependências

### Linguagem Principal
- **Python 3.10+**

### Bibliotecas Principais

| Categoria | Biblioteca | Uso |
|-----------|-----------|-----|
| DoE / Amostragem | `pyDOE2`, `SALib` | LHS, Sobol, planejamento fatorial |
| Machine Learning | `scikit-learn` | Random Forest, validação, métricas |
| Deep Learning | `PyTorch` / `TensorFlow` | Redes Neurais Artificiais (ANN) |
| Metamodelagem | `SMT` (Surrogate Modeling Toolbox) | Kriging, RBF, comparação de surrogate models |
| XAI — SHAP | `shap` | Shapley Additive Explanations |
| XAI — PDP/ICE | `scikit-learn` | Partial Dependence Plots, ICE |
| Dados | `pandas`, `numpy` | Manipulação e processamento |
| Visualização | `matplotlib`, `seaborn`, `plotly` | Gráficos e dashboards |
| Experimentos | `mlflow` | Rastreabilidade e versionamento de artefatos |
| Testes | `pytest` | Testes unitários e de integração |

### Software Externo
- **ProModel** — Simulador de Eventos Discretos (licença necessária)

---

## Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/<usuario>/tcc-xai-metamodeling.git
cd tcc-xai-metamodeling
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure os parâmetros do experimento

Edite o arquivo `config.yaml` com os parâmetros do seu modelo SED:

```yaml
doe:
  method: lhs          # lhs | sobol | full_factorial
  n_samples: 200
  random_seed: 42

simulation:
  promodel_path: "C:/ProModel/..."
  model_file: "path/to/model.mod"

metamodeling:
  algorithms: [random_forest, ann, gradient_boosting]
  test_size: 0.2
  cv_folds: 5

xai:
  methods: [shap, pdp, ice]
  shap_n_samples: 100
```

---

## Como Usar

### Execução completa do pipeline

```bash
python main.py --config config.yaml
```

### Execução modular (por etapa)

```bash
# 1. Gerar plano amostral
python doe/sampling.py --config config.yaml

# 2. Executar simulações no ProModel (requer ProModel instalado)
python simulation/promodel_interface.py --config config.yaml

# 3. Treinar e validar metamodelos
python metamodeling/trainer.py --data data/processed/dataset.csv

# 4. Aplicar camada XAI sobre o melhor metamodelo
python xai/shap_analysis.py --model results/best_model.pkl
```

### Notebooks interativos

Para exploração e análise visual, utilize os notebooks em `notebooks/`.

---

## Módulos

### `doe/` — Design of Experiments

Responsável pela geração de planos amostrais que cobrem o espaço paramétrico de forma estatisticamente representativa.

- **Métodos suportados:** Latin Hypercube Sampling (LHS), Sobol Sequence, Full Factorial
- **Entrada:** definição das variáveis de entrada e seus intervalos
- **Saída:** matriz de pontos amostrais `X` para execução no ProModel

### `metamodeling/` — Metamodelagem

Treina e compara arquiteturas de surrogate models sobre os dados coletados.

- **Modelos suportados:** Random Forest, ANN, Gradient Boosting, Kriging/GP
- **Entrada:** dataset `(X, y)` de entrada-saída do simulador
- **Saída:** modelos treinados, métricas de fidelidade, log de execução

### `xai/` — Camada de Explicabilidade

Aplica técnicas de XAI *post-hoc* sobre o metamodelo selecionado.

- **SHAP:** importâncias globais e explicações locais por instância
- **PDP:** efeito marginal médio de cada variável na saída
- **ICE:** curvas individuais de dependência condicional
- **Saída:** visualizações exportadas em `results/plots/` + relatório de sensibilidade

---

## Métricas de Validação

O framework reporta automaticamente as seguintes métricas para cada metamodelo treinado:

| Métrica | Descrição |
|---------|-----------|
| **R²** | Coeficiente de determinação — aderência geral ao simulador |
| **RMSE** | Raiz do erro quadrático médio |
| **MAE** | Erro absoluto médio |
| **Speedup** | Razão entre tempo de inferência do metamodelo e tempo de simulação equivalente |
| **Config. Accuracy** | Capacidade de identificar a configuração ótima dentre os cenários avaliados |

---

## Cronograma

| Etapa | Descrição | Período |
|-------|-----------|---------|
| 1 | Revisão bibliográfica | Mar – Mai 2026 |
| 2 | Estudo do ProModel | Mar – Abr 2026 |
| 3 | Planejamento do DoE | Abr – Mai 2026 |
| 4 | Arquitetura do framework | Mai – Jun 2026 |
| 5 | Escrita (entregas parciais) | Mar – Jun 2026 |
| — | **Entrega Monografia 1º Sem.** | **Jun 2026** |
| 6 | Modelos sintéticos e DoE | Jul – Ago 2026 |
| 7 | Desenvolvimento do framework | Jul – Set 2026 |
| 8 | Integração da camada XAI | Ago – Set 2026 |
| 9 | Validação e benchmarking | Set – Out 2026 |
| 10 | Aplicação ao caso real | Ago – Out 2026 |
| 11 | Escrita final e apresentação | Out – Nov 2026 |
| — | **Entrega Final** | **Nov 2026** |

---

## Autores

| Nome | GitHub | Contato |
|------|--------|---------|
| **Mateus Matzkin Gurza** | [@...](https://github.com) | — |
| **Vinicius Barile Lora Franco** | [@...](https://github.com) | — |

---

## Orientador

**Prof. Fabrício Junqueira**  
Escola Politécnica da Universidade de São Paulo (POLI-USP)  
Departamento de Engenharia Mecatrônica

---

## Referências

As referências completas estão em [`monografia/refs.bib`](monografia/refs.bib). Principais trabalhos que embasam o projeto:

- SAVES, P. Interpretable and explainable surrogate modeling for simulations: A state-of-the-art survey and perspectives on explainable AI for decision-making. *arXiv preprint*, 2025.

- SAVES, P. et al. SMT 2.0: A surrogate modeling toolbox with a focus on hierarchical and mixed variables Gaussian processes. *Advances in Engineering Software*, v. 188, p. 103571, 2024.

- NAGAHISARCHOGHAEI, M. et al. An empirical survey on explainable AI technologies: Recent trends, use-cases, and categories from technical and application perspectives. *Electronics*, v. 12, n. 5, p. 1092, 2023.

- MONTEVECHI, J. A. B. et al. Ensemble-based infill search simulation optimization framework. *Proceedings of the 2023 Winter Simulation Conference (WSC)*, p. 3059–3070, 2023.

- FORRESTER, A. I. J.; SÓBESTER, A.; KEANE, A. J. *Engineering Design via Surrogate Modelling: A Practical Guide*. John Wiley & Sons, 2008.

---

> *Este repositório é parte integrante do Trabalho de Conclusão de Curso submetido à Escola Politécnica da USP como requisito parcial para obtenção do título de Engenheiro de Produção.*
