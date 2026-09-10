"""Fase 3 — metamodelos.

Quatro famílias, um arquivo cada, INDEPENDENTES entre si (nenhum importa o
outro):

    baseline.py       DummyRegressor, LinearRegression, polinomial, Ridge,
                      polinomial regularizado — o piso de comparação
    tree_family.py    RandomForest, ExtraTrees, GradientBoosting, XGBoost —
                      aceitam sample_weight, compatíveis com TreeSHAP
    gp_family.py      hetGPy, smt.KRG, GaussianProcessRegressor —
                      variância preditiva nativa
    neural_family.py  MLPRegressor (e só; ver a nota sobre PyTorch lá)

NENHUM ALGORITMO É IMPLEMENTADO AQUI. Todas as famílias já existem em
bibliotecas maduras; o trabalho deste pacote é agrupá-las por família e
padronizar o mínimo para que treinar e tunar cada variante não exija
lembrar de uma assinatura diferente.

O CONTRATO, e ele é curto — em cada arquivo, para cada variante:

    build_<variante>(...)   -> estimador configurado com defaults sensatos
                               para o nosso regime (poucas centenas de
                               pontos, ruído alto)
    GRID_<VARIANTE>         -> grid de hiperparâmetros sugerido

Todos os estimadores seguem a API do scikit-learn (fit/predict), então
GridSearchCV e cross_val_score funcionam sem adaptação. hetGPy e smt.KRG
não seguem essa API nativamente e por isso têm wrappers finos
(HetGPRegressor, SMTKrigingRegressor) — compatibilidade apenas, sem lógica
de modelo.

Cada arquivo traz ainda fit_with_sample_weight(), cópia deliberadamente
idêntica nos quatro para manter a independência entre eles.

NÃO EXISTE aqui — e é decisão, não omissão: Protocol/ABC compartilhado,
registry de nomes, camada de abstração de incerteza, nem trainer
orquestrador. Isso seria infraestrutura para uma intercambiabilidade em
runtime que este projeto não precisa: quem chama sabe qual família quer,
importa aquele arquivo e usa.

USO — o laço por KPI é de quem chama, não deste pacote:

    from sed_meta.metamodels.tree_family import (
        build_random_forest, GRID_RANDOM_FOREST, fit_with_sample_weight,
    )

    X_train = pipe.transform_X(train_df)
    for kpi in kpis:                       # um modelo por KPI: X é
        y_train = pipe.transform_y(train_df, kpi)   # compartilhado, y não
        modelo = build_random_forest()
        fit_with_sample_weight(modelo, X_train, y_train,
                               sample_weight=train_df[f"{kpi}_sample_weight"])

FRONTEIRAS DE FASE: os modelos treinam e preveem no espaço TRANSFORMADO
pelo preprocessing — inverse_transform_y é da Fase 4. Métricas (R², RMSE,
comparação com o teto de ruído) são Fase 4. SHAP/PDP/Sobol são Fase 5.
"""
