from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# algorithm mappings
alg_map = {
    "lr": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic": ElasticNet,
    "dtreg": DecisionTreeRegressor,
    "rfreg": RandomForestRegressor,
    "abreg": AdaBoostRegressor,
    "gbreg": GradientBoostingRegressor
}