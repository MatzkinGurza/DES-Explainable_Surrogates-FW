"""
Metamodelos, um arquivo por família:

    baseline.py       média, linear, polinomial, Ridge e polinomial regularizado
    tree_family.py    Random Forest, Extra Trees, Gradient Boosting, XGBoost
    gp_family.py      processo gaussiano (scikit-learn), hetGP e Kriging do SMT
    neural_family.py  MLP

Em cada arquivo, para cada variante: build_<variante>() devolve o estimador
com a API do scikit-learn (fit/predict) e GRID_<VARIANTE> é a grade de
hiperparâmetros usada na busca. CATALOGO reúne as variantes comparadas na
monografia; em framework.ipynb, trocar MODELO troca a família.
"""

from modulos.metamodels import baseline, gp_family, neural_family, tree_family

#: nome -> (função que constrói o estimador, grade de hiperparâmetros)
CATALOGO = {
    "dummy": (baseline.build_dummy_mean, baseline.GRID_DUMMY_MEAN),
    "poly_ridge": (baseline.build_polynomial_ridge, baseline.GRID_POLYNOMIAL_RIDGE),
    "random_forest": (tree_family.build_random_forest, tree_family.GRID_RANDOM_FOREST),
    "extra_trees": (tree_family.build_extra_trees, tree_family.GRID_EXTRA_TREES),
    "gradient_boosting": (tree_family.build_gradient_boosting, tree_family.GRID_GRADIENT_BOOSTING),
    "xgboost": (tree_family.build_xgboost, tree_family.GRID_XGBOOST),
    "gp": (gp_family.build_sklearn_gp, gp_family.GRID_SKLEARN_GP),
    "mlp": (neural_family.build_mlp, neural_family.GRID_MLP),
}
