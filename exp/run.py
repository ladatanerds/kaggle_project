import pandas as pd
import json
import warnings
from exp.hyp.search import random_search, grid_search
from exp.train import train_model
from exp.mappings import alg_map
warnings.filterwarnings("ignore")


def run_experiment(X, Y, alg, alg_params, n_fold=5, X_test=None, score_df=None, save_results=None, search_type="random",
                   num_searches=100):
    """
    This runs a hyper-parameter search experiment.

    Parameters
    ----------
    alg : str
        A string which represents the algorithm to do hyper-parameter search for.
    alg_params: dict
        A dictionary, where the key is the hyper-parameter, and the value is a list of possible hyper-parameter values.
    score_df : Pandas.DataFrame or None
        If None, an empty DataFrame is created with "alg", "score", "mad", and "params_json" columns. Otherwise, a
        DataFrame of scores from previous experiment(s) is added to during the experiment
    search_type : str
        Choices are `random` or `grid`, representing random search and grid search respectively.
    num_searches : int
        The number of hyper-parameter searches (only applicable for random search)

    Returns
    -------
    Pandas.DataFrame
        A DataFrame of scores from the current experiment potentially with scores from previous experiment(s).
    """
    if score_df is None:
        score_df = pd.DataFrame({}, columns=["alg", "score", "mad", "params_json"])
    elif isinstance(score_df, str):
        score_df = pd.read_csv(score_df)
    # different search options to generate hyper-parameter experiments
    if search_type == "random":
        param_searches = random_search(num_searches=num_searches, **alg_params)
    if search_type == "grid":
        param_searches = grid_search(**alg_params)

    # retrieve algorithm
    alg_cls = alg_map[alg]

    # run experiment
    for param_search in param_searches:
        # instantiate model from hyper-parameters
        model = alg_cls(**param_search)
        # produce cv score and mad
        score, mad = train_model(X=X, Y=Y, X_test=X_test, n_fold=n_fold, params=None, model=model)
        # generate dataframe row to track alg scores
        df_ = pd.DataFrame(
            {"alg": [alg], "score": [score], "mad": [mad], "params_json": [json.dumps(param_search, sort_keys=True)]})
        # append to overall dataframe
        score_df = score_df.append(df_)
        # save to csv file after each search completes
        if isinstance(save_results, str):
            score_df.to_csv(save_results, index=False)
    return score_df
