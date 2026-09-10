# DES-Explainable_Surrogates-FW

# Framework de Metamodelagem Explicável para Análise Paramétrica de Sistemas de Eventos Discretos

TCC — Escola Politécnica da Universidade de São Paulo (POLI-USP)
Departamento de Engenharia Mecatrônica e de Sistemas Mecânicos (PMR)
Mateus Matzkin Gurza · Vinicius Barile Lora Franco

---

## 📋 Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Motivação](#motivação)
- [Objetivos](#objetivos)
- [Estado atual da implementação](#estado-atual-da-implementação)
- [Arquitetura do Framework](#arquitetura-do-framework)
- [Interface com o ProModel — o cockpit Excel](#interface-com-o-promodel--o-cockpit-excel)
- [Decisões de projeto registradas](#decisões-de-projeto-registradas)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Tecnologias e Dependências](#tecnologias-e-dependências)
- [Instalação e Configuração](#instalação-e-configuração)
- [Como Usar](#como-usar)
- [Módulos](#módulos)
- [Métricas de Validação](#métricas-de-validação)
- [Testes](#testes)
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

**Palavras-chave:** Simulação de Eventos Discretos · SED · Metamodelagem · Surrogate Models · Explicabilidade · XAI · ProModel · Projeto de Experimentos · DoE · SHAP · Random Forest · Processos Gaussianos

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
2. **Comparação sistemática** de arquiteturas de metamodelagem (baseline, árvores, Processos Gaussianos, redes neurais) em termos de acurácia, generalização, tempo de treinamento, latência de inferência e custo computacional.
3. **Conjunto de métricas de fidelidade** que quantificam o grau de aderência do metamodelo ao simulador de referência — incluindo a comparação contra o **teto de ruído** do próprio simulador.
4. **Relatórios de explicabilidade** que correlacionam as importâncias de variáveis extraídas via XAI com a teoria de modelos analíticos de sistemas produtivos e logísticos.
5. **Framework aplicado** em pelo menos dois sistemas: modelos sintéticos de referência e um caso industrial real (dados anonimizados).

### Testes para Validação

| # | Critério | Métrica |
|---|----------|---------|
| 1 | Taxa de acerto das variáveis de saída | R², RMSE, MAE — sempre lidos contra o teto de ruído |
| 2 | Taxa de acerto da configuração ótima | Identificação da melhor configuração paramétrica |
| 3 | Tempo de inferência | Razão entre tempo do metamodelo e tempo do simulador |
| 4 | Testes de explicabilidade | Confirmação do efeito de cada variável via replicação no ProModel |

---

## Estado atual da implementação

O framework segue o fluxo metodológico da monografia, dividido em fases. O que já existe em código:

| Fase | Pacote | Estado | Conteúdo |
|------|--------|--------|----------|
| 1 — DoE | `doe/parameter_space.py` | ✅ Implementado | Espaço de parâmetros misto e geração de design one-shot (LHS/Monte Carlo/Sobol) |
| 1 — Replicações | `doe/replication.py` | ✅ Implementado | Política fixa de replicação, plano de execução, agregação e **teto de ruído** |
| 2 — Simulador | `sim_interface/excel_bridge.py` | ✅ Implementado | Cockpit `.xlsx` de ida e volta com o ProModel, com esquema de tipos e status por execução |
| 2→3 — Ponte | `preprocessing/pipeline.py` | ✅ Implementado | Split / fit / transform de X e y com separação estrita entre transformação determinística e aprendida |
| 3 — Metamodelos | `metamodels/` | ✅ Implementado | Quatro famílias: `baseline`, `tree_family`, `gp_family`, `neural_family` |
| 4 — Avaliação | `evaluation/` | 🚧 Esqueleto | Métricas de erro, validação cruzada, leitura contra o teto de ruído |
| 5 — XAI | `xai/` | 🚧 Esqueleto | SHAP, PDP/ICE, índices de Sobol, incerteza via conformal prediction |
| 6 — Validação analítica | `theoretical/` | 📝 Previsto | Confronto das importâncias XAI com modelos analíticos |
| — Orquestração | `cycle/` | 🚧 Esqueleto | Controlador dos Fluxos 1/2/3 |
| — Rastreabilidade | `logging_/` | 🚧 Esqueleto | Logging estruturado e tracking de execuções (structlog/MLflow) |

Ainda **não** existem: `main.py`/CLIs em `scripts/`, notebooks e os YAMLs de cenário preenchidos (os quatro arquivos em `configs/` estão com o esqueleto de campos, aguardando os parâmetros dos modelos ProModel).

---

## Arquitetura do Framework

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             FRAMEWORK PIPELINE                               │
│                                                                              │
│  Fase 1                     Fase 2                    Fase 3                 │
│  ┌────────────────┐   plano  ┌──────────────────┐  raw  ┌─────────────────┐  │
│  │ parameter_space│ ───────► │  cockpit .xlsx   │ ────► │ preprocessing   │  │
│  │ + replication  │  N x R   │  <->  ProModel   │       │  X / y          │  │
│  └────────────────┘  linhas  └──────────────────┘       └────────┬────────┘  │
│         ▲                                                        │           │
│   ParameterSpec                                                  ▼           │
│   (configs/*.yaml)                                      ┌─────────────────┐  │
│                                                         │  metamodels     │  │
│                          teto de ruído  ─────────────►  │  4 famílias     │  │
│                          (compute_noise_ceiling)        └────────┬────────┘  │
│                                                                  │           │
│                             Fase 4                               ▼           │
│                          ┌─────────────────┐          ┌─────────────────┐    │
│                          │  evaluation     │ ───────► │   xai (Fase 5)  │    │
│                          │  R², RMSE, CV   │          │  SHAP/PDP/ICE   │    │
│                          └─────────────────┘          └─────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

O caminho completo dos dados, função por função:

```
parameter_space.generate_design()      -> N pontos do DoE
replication.build_execution_plan()     -> N x R linhas (uma por execução)
excel_bridge.build_cockpit_table()     -> + colunas de KPI vazias
excel_bridge.write_cockpit_excel()     -> .xlsx entregue ao ProModel
........... ProModel roda e preenche a planilha ...........
excel_bridge.read_cockpit_excel()      -> .xlsx de volta em DataFrame
excel_bridge.extract_results()         -> TODAS as linhas + status_execucao
excel_bridge.select_complete_runs()    -> recorte explícito das completas
replication.aggregate_replications()   -> dataset agregado (1 linha/ponto)
replication.compute_noise_ceiling()    -> teto de R² por KPI
PreprocessingPipeline.split/fit/...    -> X_train/X_test, y_train/y_test
metamodels.build_*()                   -> estimador treinado
```

---

## Interface com o ProModel — o cockpit Excel

Esta é a diferença arquitetural mais importante em relação ao esboço inicial do projeto: **o Python não automatiza o ProModel**. A troca é feita por uma planilha `.xlsx` ("cockpit") no formato

```
uma linha = uma execução completa (um ponto do DoE x uma replicação)

[ ------------- ENTRADA -------------- ] [ ---- SAÍDA ---- ]
point_id | replication_id | seed | <params...> | <KPIs...>
                                                  (vazios)
```

O ProModel é configurado **por fora** deste código para varrer a planilha sequencialmente: pega a próxima linha ainda não executada, lê os parâmetros daquela linha, roda a simulação e escreve os KPIs na **mesma linha** antes de avançar. Célula em branco é o marcador de "ainda não executado".

Dois detalhes de formato que valem registro:

1. **Um único cabeçalho** (linha 1), sem faixa mesclada "Entrada"/"Saída" acima — uma segunda linha de cabeçalho deslocaria os dados e tornaria a leitura pelo ProModel ambígua. A distinção visual é feita apenas por cor de preenchimento e negrito nas células do cabeçalho.
2. **Aba oculta de esquema** (`_sed_meta_schema`) com o dtype original de cada coluna. Sem ela o round-trip pelo Excel perde tipo: `int64` volta como `float64`, `category` volta como texto solto e a ordem de uma categórica ordenada (ex.: prioridade `baixa < media < alta`) desaparece. Se a aba não existir, a leitura cai num modo *best-effort* documentado.

---

## Decisões de projeto registradas

Quatro decisões estruturam o código e explicam por que ele é do jeito que é.

### 1. Nada é deletado silenciosamente

Nenhuma linha desaparece em nenhum ponto do pipeline. `extract_results()` devolve **todas** as execuções do cockpit e marca cada uma na coluna `status_execucao`:

| Status | Significado |
|--------|-------------|
| `completa` | todos os KPIs preenchidos e numéricos |
| `parcial` | alguns KPIs preenchidos, outros em branco |
| `pendente` | nenhum KPI preenchido (ainda não rodou) |
| `invalida` | KPI preenchido com algo que não é número |

A seleção das completas é um passo **explícito e separado** (`select_complete_runs()`), que loga o que está deixando de fora.

### 2. Entradas determinísticas, saídas aprendidas

A regra que organiza o `preprocessing` inteiro, e o erro que ele existe para tornar impossível:

- **X (entradas)** — os limites de um parâmetro são conhecidos *antes* de qualquer simulação rodar: vêm do `ParameterSpec` (o YAML do cenário), não dos dados coletados. Por isso **não** se usa `MinMaxScaler().fit(X_train)`, e sim os limites *declarados*; idem para categóricas, com `OneHotEncoder(categories=[spec.levels])` explícito. Propriedade garantida e testada: ajustar o transformador de X em **qualquer** subconjunto das linhas — inclusive uma única — produz exatamente a mesma transformação.
- **y (saídas)** — o range de um KPI só existe depois de rodar simulação. Aqui a transformação é ajustada aos dados, e a ordem é crítica: `split()` nos dados brutos → `fit()` **só** no treino → `transform_y()` (nunca `fit_transform`) nos dois lados.

### 3. O teto de ruído é o denominador de tudo

`compute_noise_ceiling()` decompõe a variância observada em variância *entre* pontos (sinal) e *intra* ponto (ruído do simulador):

```
noise_ceiling = variância_entre_pontos / variância_total
```

Nenhum metamodelo pode ter R² acima desse valor de forma legítima — se tiver, é overfitting ao ruído. Um R² de 0,72 num KPI cujo teto é 0,75 é um resultado **excelente**; o mesmo 0,72 num KPI de teto 0,98 é um metamodelo ruim. É por isso que as métricas da Fase 4 nunca são lidas em absoluto.

### 4. Escopo fechado em duas frentes

- **Sem PyTorch.** A família neural é `MLPRegressor` e só. Redes neurais rendem com muitos dados, e o regime aqui é de poucas centenas de pontos de DoE, cada um custando uma rodada de ProModel com replicações. O gatilho para reconsiderar é único e ainda não observado: o MLP do sklearn se mostrar competitivo *e* limitado pela própria implementação.
- **Nenhum algoritmo é implementado do zero.** Todas as famílias já existem em bibliotecas maduras; o trabalho do pacote `metamodels` é agrupá-las por família e padronizar o mínimo — `build_<variante>()` + `GRID_<VARIANTE>` — para que treinar e tunar cada variante não exija lembrar de uma assinatura diferente.

---

## Estrutura do Repositório

```
📦 DES-Explainable_Surrogates-FW/
│
├── 📂 Código/
│   ├── 📂 configs/                      # Cenários (validados via pydantic)
│   │   ├── scenario_01_serial.yaml      #   linha serial (baseline)
│   │   ├── scenario_02_routing.yaml     #   roteamento
│   │   ├── scenario_03_multiproduto.yaml
│   │   └── scenario_real.yaml           #   estudo de caso real
│   │
│   ├── 📂 src/sed_meta/                 # Pacote principal (layout src/)
│   │   ├── doe/                         # FASE 1
│   │   │   ├── parameter_space.py       #   ParameterSpec, generate_design
│   │   │   └── replication.py           #   ReplicationPolicy, agregação, teto de ruído
│   │   ├── sim_interface/               # FASE 2
│   │   │   └── excel_bridge.py          #   cockpit .xlsx <-> ProModel
│   │   ├── preprocessing/               # PONTE FASE 2 -> 3
│   │   │   └── pipeline.py              #   PreprocessingPipeline, TargetTransformSpec
│   │   ├── metamodels/                  # FASE 3 — quatro famílias independentes
│   │   │   ├── baseline.py              #   Dummy, Linear, polinomial, Ridge
│   │   │   ├── tree_family.py           #   RF, ExtraTrees, GradientBoosting, XGBoost
│   │   │   ├── gp_family.py             #   sklearn GP, smt.KRG, hetGPy
│   │   │   └── neural_family.py         #   MLPRegressor
│   │   ├── evaluation/                  # FASE 4 (esqueleto)
│   │   ├── xai/                         # FASE 5 (esqueleto)
│   │   ├── cycle/                       # Controlador dos Fluxos 1/2/3 (esqueleto)
│   │   └── logging_/                    # RNF de rastreabilidade (esqueleto)
│   │
│   ├── 📂 tests/
│   │   ├── metamodels/                  # conftest, helpers, baseline, tree_family
│   │   ├── preprocessing/               # invariância do transformador de X, vazamento em y
│   │   └── test_excel_bridge_e2e.py     # round-trip completo do cockpit
│   │
│   ├── 📂 notebooks/                    # exploração (vazio por ora)
│   ├── 📂 scripts/                      # CLIs finos com click (vazio por ora)
│   ├── 📂 runs/                         # saída por execução — NÃO versionado
│   ├── pytest.ini
│   └── requirements.txt
│
├── 📂 Monografia/                       # LaTeX — só o PDF é versionado
│   └── main.pdf
│
├── .gitignore
└── README.md
```

> **Sobre o versionamento:** o `.gitignore` fica na raiz e cobre `venv/`, `runs/`, `mlruns/` e os caches (`__pycache__`, `.pytest_cache`, `.ipynb_checkpoints`). De `Monografia/` **apenas `main.pdf`** é versionado — as fontes `.tex`, o `refs.bib`, `images/` e todos os artefatos de compilação do LaTeX ficam de fora.

---

## Tecnologias e Dependências

### Linguagem Principal

- **Python 3.12** (testado nessa versão; requer 3.10+ pela sintaxe `X | None`)

### Bibliotecas Principais

| Categoria | Biblioteca | Uso |
|-----------|-----------|-----|
| DoE / Amostragem | `smt>=2.14` | `DesignSpace` misto + `LHS(criterion="ese")`, `MixedIntegerSamplingMethod` |
| DoE / Amostragem | `scipy>=1.13` | `scipy.stats.qmc`: LatinHypercube, Sobol |
| DoE / Sensibilidade | `SALib>=1.5` | Índices de Sobol (triangulação independente do SMT-EX) |
| DoE / Screening | `pyDOE3>=1.6` | Fatorial fracionado, Plackett-Burman |
| Interface simulador | `openpyxl>=3.1` | Leitura/escrita do cockpit `.xlsx` |
| Pré-processamento | `scikit-learn>=1.5` | `ColumnTransformer`, `OneHotEncoder` + metamodelos e CV |
| Pré-processamento | `joblib>=1.4` | Persistência do pipeline ajustado em `runs/<run_id>/` |
| Metamodelagem — árvores | `xgboost>=3.4` | GBT |
| Metamodelagem — GP | `hetgpy>=1.0` | Processo Gaussiano **heterocedástico** |
| XAI | `smt-explainability>=0.3` | SHAP/PDP/ICE/Sobol integrados ao SMT |
| XAI | `shap>=0.52` | TreeExplainer / KernelExplainer (fonte independente) |
| XAI — incerteza | `mapie>=1.5` | Conformal prediction, para modelos sem UQ nativa |
| Configuração | `pydantic>=2.13`, `PyYAML>=6.0` | Validação de schema dos YAMLs de cenário |
| Rastreabilidade | `mlflow>=3.16`, `structlog>=25.1` | Tracking por execução, logging estruturado |
| Visualização | `matplotlib>=3.9`, `seaborn>=0.13` | Relatórios, PDP/ICE, figuras da monografia |
| CLI | `click>=8.1` | `scripts/run_*.py` como CLIs finos |
| Testes | `pytest>=8.3`, `pytest-cov>=5.0` | Testes unitários e de integração |
| Notebooks | `jupyter>=1.1`, `ipykernel>=6.29` | Exploração |

> Decisões de dependência que mudaram em relação ao esboço inicial: `pyDOE3` no lugar de `pyDOE2`; **sem PyTorch/TensorFlow** (ver [Decisões de projeto](#4-escopo-fechado-em-duas-frentes)); `hetgpy` acrescentado por ser o GP tecnicamente adequado a SED estocástico; `mapie` acrescentado para dar incerteza às famílias que não têm UQ nativa.

### Software Externo

- **ProModel** — Simulador de Eventos Discretos (licença necessária). Configurado manualmente para consumir o cockpit `.xlsx`; a configuração do ProModel está fora do escopo do código Python.

---

## Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/MatzkinGurza/DES-Explainable_Surrogates-FW.git
cd DES-Explainable_Surrogates-FW
```

### 2. Crie e ative um ambiente virtual

O `venv/` vive na **raiz** do repositório (um nível acima de `Código/`) e não é versionado.

```bash
python -m venv venv
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r Código/requirements.txt
```

### 4. Configure o cenário

Edite o YAML do cenário desejado em `Código/configs/`. O esqueleto de campos é:

```yaml
scenario:
  name: scenario_01_serial
  description: "Linha de produção serial (baseline)"

factors: []        # fatores de entrada e seus intervalos (viram ParameterSpec)
responses: []      # KPIs de interesse (viram as colunas de saída do cockpit)
replications: 1    # replicações por ponto de projeto
```

---

## Como Usar

Ainda não existem os CLIs em `scripts/` — o uso atual é via API do pacote. O layout `src/` está resolvido pelo `pytest.ini` (`pythonpath = src`) para os testes; para uso interativo, rode a partir de `Código/` com `PYTHONPATH=src`.

### Fase 1 → 2: gerar o design e emitir o cockpit

```python
from sed_meta.doe.parameter_space import ParameterSpec, generate_design
from sed_meta.doe.replication import ReplicationPolicy, build_execution_plan
from sed_meta.sim_interface.excel_bridge import build_cockpit_table, write_cockpit_excel

specs = [
    ParameterSpec("n_operadores",  "integer",     low=2, high=10),
    ParameterSpec("taxa_chegada",  "float",       low=0.5, high=4.0),
    ParameterSpec("regra_fila",    "categorical", levels=["FIFO", "SPT", "EDD"]),
    ParameterSpec("prioridade",    "ordinal",     levels=["baixa", "media", "alta"]),
]

design = generate_design(specs, n_points=120, seed=42, method="lhs", criterion="ese")
plan   = build_execution_plan(design, ReplicationPolicy(n_replications=5, seed_base=0))

cockpit = build_cockpit_table(plan, kpi_names=["lead_time", "wip", "utilizacao"])
write_cockpit_excel(cockpit, "runs/exp001/cockpit.xlsx")
```

Entregue o `.xlsx` ao ProModel. Ele preenche as colunas de KPI linha a linha.

### Fase 2 → 3: ler de volta, agregar e treinar

```python
from sed_meta.sim_interface.excel_bridge import (
    read_cockpit_excel, extract_results, select_complete_runs,
)
from sed_meta.doe.replication import aggregate_replications, compute_noise_ceiling
from sed_meta.preprocessing import PreprocessingPipeline, TargetTransformSpec
from sed_meta.metamodels.tree_family import build_random_forest, fit_with_sample_weight

kpis = ["lead_time", "wip", "utilizacao"]

df   = read_cockpit_excel("runs/exp001/cockpit.xlsx")
raw  = extract_results(df, kpi_names=kpis)      # TODAS as linhas + status_execucao
done = select_complete_runs(raw)                # recorte explícito, com log

param_cols = [s.name for s in specs]
dataset = aggregate_replications(done, param_cols=param_cols, kpi_cols=kpis)
teto    = compute_noise_ceiling(done, kpi_cols=kpis)   # R² máximo legítimo por KPI

pipe = PreprocessingPipeline(
    param_specs=specs,
    target_specs=[TargetTransformSpec(name=k, method="standard") for k in kpis],
)
train_df, test_df = pipe.split(dataset, test_size=0.2, seed=42)
pipe.fit(train_df)                              # SÓ o treino

X_train, X_test = pipe.transform_X(train_df), pipe.transform_X(test_df)
y_train = pipe.transform_y(train_df, "lead_time")

model = build_random_forest()
fit_with_sample_weight(model, X_train, y_train,
                       sample_weight=train_df["lead_time_sample_weight"])

# para reportar em minutos/peças:
y_pred = pipe.inverse_transform_y(model.predict(X_test), "lead_time")
```

**Um modelo por KPI:** `X` é compartilhado entre todos os KPIs, `y` não — por isso `transform_y`/`inverse_transform_y` recebem o nome do KPI, e o laço por KPI é de quem chama.

---

## Módulos

### `doe/` — Fase 1: Projeto de Experimentos

**`parameter_space.py`** — dado um espaço de parâmetros e um número de pontos, gera um `DataFrame` cobrindo o espaço, de forma **one-shot** (uma chamada, sem realimentação). Não sabe nada sobre replicações, warm-up ou o simulador.

- Tipos de parâmetro: `float`, `integer`, `categorical` (nominal), `ordinal` (ordem importa, espaçamento não uniforme) e `binary`. A distinção `ordinal` × `categorical` × `integer` é explícita e afeta tanto a amostragem quanto o encoding no `preprocessing`.
- Métodos: `lhs` (padrão, com critério `ese` — o mesmo de Saves et al. 2025), `monte_carlo`, `sobol`. Sobol é rejeitado com erro claro quando há categóricas nominais, por não ter definição nativa para elas.
- Saída já no dtype correto (`int64`/`float64`/`category`), com a ordem preservada nas ordinais.

**`replication.py`** — política **fixa** de replicação (todo ponto roda o mesmo número de réplicas, sem lógica adaptativa por variância). É puro e sem I/O: não roda simulação nenhuma, o que permite testar a agregação com `DataFrame` sintético.

- `build_execution_plan()` expande N pontos em N × R linhas, uma por execução, com `seed` por réplica.
- `aggregate_replications()` consolida em uma linha por ponto, com média, desvio, SEM e `sample_weight` (inverso da variância) por KPI — o gancho para metamodelos ponderados.
- `compute_noise_ceiling()` devolve o R² máximo legítimo por KPI.

### `sim_interface/` — Fase 2: interface com o ProModel

Ver [Interface com o ProModel](#interface-com-o-promodel--o-cockpit-excel). API: `build_cockpit_table`, `write_cockpit_excel`, `read_cockpit_excel`, `extract_results`, `select_complete_runs`.

### `preprocessing/` — ponte Fase 2 → 3

`PreprocessingPipeline` (split → fit → transform), `TargetTransformSpec` (métodos `standard`, `log_standard`, `none`) e `build_input_transformer`. Ver [Decisão 2](#2-entradas-determinísticas-saídas-aprendidas). O `split()` devolve `DataFrame`s **completos** — as colunas `{kpi}_std`, `{kpi}_n`, `{kpi}_sem` e `{kpi}_sample_weight` seguem disponíveis para a Fase 3.

### `metamodels/` — Fase 3: quatro famílias

Um arquivo por família, **independentes entre si** (nenhum importa o outro). Contrato uniforme: `build_<variante>()` devolve um estimador com defaults sensatos para o nosso regime (poucas centenas de pontos, ruído alto), e `GRID_<VARIANTE>` traz o grid de hiperparâmetros sugerido. Todos seguem a API do scikit-learn, então `GridSearchCV` e `cross_val_score` funcionam sem adaptação.

| Arquivo | Variantes | Por que a família é coesa |
|---------|-----------|---------------------------|
| `baseline.py` | `dummy_mean`, `linear`, `polynomial`, `ridge`, `polynomial_ridge` | O piso de comparação — nunca deve ser pulado |
| `tree_family.py` | `random_forest`, `extra_trees`, `gradient_boosting`, `xgboost` | Aceitam `sample_weight` com a mesma semântica e são compatíveis com **TreeSHAP** (exato, polinomial) |
| `gp_family.py` | `sklearn_gp`, `smt_kriging`, `hetgp` | **Variância preditiva nativa** — as outras famílias precisam de conformal prediction para dizer o mesmo |
| `neural_family.py` | `mlp` | `MLPRegressor` e só ([decisão registrada](#4-escopo-fechado-em-duas-frentes)) |

Dois pontos que valem destaque:

- **O baseline é resultado, não fracasso.** Se uma regressão polinomial de 2ª ordem já resolve, isso significa que a superfície de resposta do sistema simulado é suave e de baixa ordem — uma conclusão publicável sobre o **sistema**, mais informativa que um Random Forest opaco com o mesmo R².
- **A família GP é uma escada deliberada sobre o ruído:** `GaussianProcessRegressor` (homocedástico, via `WhiteKernel`) → `smt.KRG` (homocedástico, via `eval_noise`) → `hetGPy.hetGP` (**heterocedástico**, ruído modelado como processo). A última é a tecnicamente adequada para SED — a variância de um KPI de fila explode perto da saturação —, e as três estão presentes para que a monografia possa **medir** essa diferença em vez de afirmá-la. `hetGPy` e `smt.KRG` não seguem a API do sklearn nativamente e por isso têm wrappers finos (`HetGPRegressor`, `SMTKrigingRegressor`) — compatibilidade apenas, sem lógica de modelo.

**Espaço transformado:** os estimadores treinam e preveem no espaço já transformado pelo `preprocessing` (X escalado para [0,1] pelos limites do `ParameterSpec`, y padronizado e possivelmente em `log1p`). Devolver previsão em minutos/peças é responsabilidade da Fase 4, que tem o objeto de preprocessing em mãos.

### `evaluation/`, `xai/`, `cycle/`, `logging_/`

Esqueletos com o escopo já delimitado nas docstrings. `cycle/` é onde mora a sequência completa dos Fluxos 1/2/3 — vale notar que **não existe execução em lote do lado do Python**: quem percorre as linhas rodando a simulação é o ProModel, a partir do cockpit.

---

## Métricas de Validação

| Métrica | Descrição |
|---------|-----------|
| **R²** | Coeficiente de determinação — **sempre lido contra o teto de ruído do KPI** |
| **Teto de ruído** | `var_entre_pontos / var_total` — o R² máximo legítimo, por KPI |
| **RMSE** | Raiz do erro quadrático médio |
| **MAE** | Erro absoluto médio |
| **Speedup** | Razão entre tempo de inferência do metamodelo e tempo de simulação equivalente |
| **Config. Accuracy** | Capacidade de identificar a configuração ótima dentre os cenários avaliados |
| **Cobertura do intervalo** | Calibração da incerteza (nativa nos GPs, conformal via `mapie` nas outras famílias) |

---

## Testes

```bash
cd Código
pytest                     # pythonpath=src e testpaths=tests já vêm do pytest.ini
pytest --cov=src/sed_meta  # com cobertura
```

O que os testes cobrem hoje:

- **`tests/preprocessing/test_pipeline.py`** — a propriedade central do módulo: ajustar o transformador de X em qualquer subconjunto de linhas (inclusive uma só) produz a mesma transformação; e a ausência de vazamento na transformação de y.
- **`tests/metamodels/`** — contrato `build_*`/`GRID_*` e propagação de `sample_weight` nas famílias `baseline` e `tree_family`.
- **`tests/test_excel_bridge_e2e.py`** — round-trip completo do cockpit, incluindo preservação de dtypes pela aba de esquema e a marcação de status por execução.

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

| Nome | GitHub |
|------|--------|
| **Mateus Matzkin Gurza** | [@MatzkinGurza](https://github.com/MatzkinGurza) |
| **Vinicius Barile Lora Franco** | — |

---

## Orientador

**Prof. Fabrício Junqueira**
Escola Politécnica da Universidade de São Paulo (POLI-USP)
Departamento de Engenharia Mecatrônica e de Sistemas Mecânicos (PMR)

---

## Referências

As referências completas estão em `Monografia/refs.bib` (fonte LaTeX não versionada — ver a nota em [Estrutura do Repositório](#estrutura-do-repositório); a bibliografia compilada está em [`Monografia/main.pdf`](Monografia/main.pdf)). Principais trabalhos que embasam o projeto:

- SAVES, P. Interpretable and explainable surrogate modeling for simulations: A state-of-the-art survey and perspectives on explainable AI for decision-making. *arXiv preprint*, 2025.

- SAVES, P. et al. SMT 2.0: A surrogate modeling toolbox with a focus on hierarchical and mixed variables Gaussian processes. *Advances in Engineering Software*, v. 188, p. 103571, 2024.

- NAGAHISARCHOGHAEI, M. et al. An empirical survey on explainable AI technologies: Recent trends, use-cases, and categories from technical and application perspectives. *Electronics*, v. 12, n. 5, p. 1092, 2023.

- MONTEVECHI, J. A. B. et al. Ensemble-based infill search simulation optimization framework. *Proceedings of the 2023 Winter Simulation Conference (WSC)*, p. 3059–3070, 2023.

- FORRESTER, A. I. J.; SÓBESTER, A.; KEANE, A. J. *Engineering Design via Surrogate Modelling: A Practical Guide*. John Wiley & Sons, 2008.

---

> *Este repositório é parte integrante do Trabalho de Conclusão de Curso submetido à Escola Politécnica da USP como requisito parcial para obtenção do título de Engenheiro Mecatrônico.*
