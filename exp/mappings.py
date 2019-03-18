from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, Lars, LassoLars, \
    OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# algorithm mappings
alg_map = {
    "lr": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "mtlasso": MultiTaskLasso,
    "elastic": ElasticNet,
    "lars": Lars,
    "llars": LassoLars,
    "omp": OrthogonalMatchingPursuit,
    "sgdreg": SGDRegressor,
    "pareg": PassiveAggressiveRegressor,
    "tsreg": TheilSenRegressor,
    "hreg": HuberRegressor,
    "kreg": KernelRidge,
    "dtreg": DecisionTreeRegressor,
    "rfreg": RandomForestRegressor,
    "abreg": AdaBoostRegressor,
    "gbreg": GradientBoostingRegressor,

}