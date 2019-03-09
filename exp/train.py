import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import time
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import seaborn as sns
import warnings
from statsmodels import robust
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")


def train_model(X, Y, X_test=None, n_fold=10, params=None, model_type='sklearn', plot_feature_importance=False,
                model=None):
    """Taken from the `Earthquakes FE. More features and samples` kaggle notebook"""
    oof = np.zeros(len(X))
    if X_test is not None:
        prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    folds = KFold(n_splits=n_fold, shuffle=True)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=50000, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                      verbose=10000, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)

            if X_test is not None:
                y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                         ntree_limit=model.best_ntree_limit)
            if X_test is not None:
                y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1, )
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            if X_test is not None:
                y_pred = model.predict(X_test).reshape(-1, )
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)
            y_pred_valid = model.predict(X_valid)
            if X_test is not None:
                y_pred = model.predict(X_test)
        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        if X_test is not None:
            prediction += y_pred
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    if X_test is not None:
        prediction /= n_fold

    print('CV mean score: {0:.4f}, mad: {1:.4f}.'.format(np.mean(scores), robust.mad(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
    if X_test is not None:
        return np.mean(scores), robust.mad(scores), prediction
    else:
        return np.mean(scores), robust.mad(scores)
