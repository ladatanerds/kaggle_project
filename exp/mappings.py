from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, Lars, LassoLars, \
    OrthogonalMatchingPursuit, SGDRegressor, PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


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
    "svr": SVR,
    "nsvr": NuSVR,
    "lsvr": LinearSVR,
    "knreg": KNeighborsRegressor,
    "rnreg": RadiusNeighborsRegressor,
    "gpreg": GaussianProcessRegressor,
    "plsreg": PLSRegression,
    "dtreg": DecisionTreeRegressor,
    "bagreg": BaggingRegressor,
    "rfreg": RandomForestRegressor,
    "etreg": ExtraTreesRegressor,
    "abreg": AdaBoostRegressor,
    "gbreg": GradientBoostingRegressor,
    "mlpreg": MLPRegressor
}
