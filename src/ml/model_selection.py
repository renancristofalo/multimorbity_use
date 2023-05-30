import datetime
import gc
import os.path as osp
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from tensorflow import keras
from tensorflow.keras.layers import AlphaDropout, Dense, Input
from tqdm import tqdm

from src.ml.utils.utils import write_joblib
from src.utils.utils import time_diff_str


def cross_validate_models(
    models: list,
    X_train,
    y_train,
    X_test,
    y_test,
    metrics: str = None,
    n_folds: int = 5,
    models_dir=None,
    **kwargs,
):
    """[Makes the process of Cross-Validation in the list passed]

    Args:
        models (list): [List with all models]
        X_train ([pd.DataFrame]): [The data to fit]
        y_train ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        X_test ([pd.DataFrame]): [The data to predict]
        y_test ([pd.Series]): [The target variable with the value to be predicted]
        scoring (str, optional): [Strategy to evaluate the performance of the cross-validated model on the test set]. Defaults to None.
        n_folds (int, optional): [Specify the number of folds]. Defaults to 5.

    Returns:
        [tuple]: [Return with the names,results and dataframe with the results of the models]
    """
    if isinstance(metrics, str):
        metrics = {metrics: metrics}
    elif isinstance(metrics, (list, tuple, set)):
        metrics = {metric: metric for metric in metrics}

    results, names, metric_dict = {}, [], {}

    for model_dict in tqdm(models):
        name = model_dict.get("name")
        model = model_dict.get("model")

        # Defining Metric results dict
        metric_dict[name] = _create_metric_dict(metrics)

        # makes the cross-validation process for each model
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=n_folds,  # if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
            scoring=metrics,
            n_jobs=1,
            return_train_score=True,
            return_estimator=True,
            **kwargs,
        )

        # store the results and name of the model
        names.append(name)

        # Loops through the metrics getting the test results, adding it to the metric_dict and logging.info the result
        for metric_name, scorer in metrics.items():
            scorer = get_scorer(scorer)
            test_scorer = np.array(
                [
                    scorer(model_fold, X_test, y_test)
                    for model_fold in cv_results["estimator"]
                ]
            )

            metric_dict[name][metric_name.upper()]["TRAIN"] = cv_results[
                f"train_{metric_name}"
            ]
            metric_dict[name][metric_name.upper()]["VALIDATION"] = cv_results[
                f"test_{metric_name}"
            ]
            metric_dict[name][metric_name.upper()]["TEST"] = test_scorer

        metric_dict[name]["FIT_TIME"] = cv_results["fit_time"]

        msg_metrics = _create_metric_msg(metric_dict[name])

        # print of the result of each model
        print(f"\nAVERAGES - {name.upper()}:\n" + msg_metrics)

        results.update(metric_dict)

        model.fit(X_train, y_train)

        if models_dir:
            write_joblib(model, osp.join(models_dir, name, f"model.p"))
            write_joblib(metric_dict, osp.join(models_dir, name, f"metrics.p"))
            write_joblib(cv_results, osp.join(models_dir, name, f"cv_results.p"))

            write_joblib(results, osp.join(models_dir, f"results.p"))

    df_results = create_df_results(results)

    return (names, results, df_results)

def _create_metric_dict(metrics):
    metric_dict = {}
    for metric_name in metrics.keys():
        metric_dict[metric_name.upper()] = {
            "TRAIN": np.array([]),
            "VALIDATION": np.array([]),
            "TEST": np.array([]),
        }
    return metric_dict


def _create_metric_msg(metric_dict):
    """Creates a string with the average of the metrics for each model"""
    metrics_msg = ""
    max_size = max([len(metric) for metric in metric_dict])
    for metric_name, metric_results in metric_dict.items():
        if metric_name != "FIT_TIME":
            start_msg = f" - {metric_name.upper()} {' '*(max_size-len(metric_name) + 10)}-> "
            train_msg = f" - TRAIN: {metric_results['TRAIN'].mean():.2f} (+/- {metric_results['TRAIN'].std():.2f})"
            valid_msg = f" | VALIDATION: {metric_results['VALIDATION'].mean():.2f} (+/- {metric_results['VALIDATION'].std():.2f})"
            test_msg = (
                f" | TEST: {metric_results['TEST'].mean():.2f} (+/- {metric_results['TEST'].std():.2f})\n"
                ""
            )
            metrics_msg += start_msg + train_msg + valid_msg + test_msg
    metrics_msg += f" - FIT TIME FOR EACH FOLD ->  - TRAIN: { metric_dict['FIT_TIME'].mean():.2f}s (+/- { metric_dict['FIT_TIME'].std():.2f}s)\n"
    return metrics_msg


def create_df_results(results, models=None, slim=False):
    """Creates a df with all model metrics

    Args:
        slim (bool, optional): [If the df should be slim (with the sets - Train, Validation and Test - on rows) or not]. Defaults to False.

    Returns:
        [pd.Dataframe]: [Dataframe with the model results]
    """
    # Organizes the results in a dataframe

    if not models:
        models = results.keys()

    aux_df_models = []
    for model_name in results:
        if model_name in models:
            n_folds = len(results[model_name]["FIT_TIME"])
            aux_df_metrics = []
            for metric in results[model_name]:
                if isinstance(results[model_name][metric], dict):

                    aux_df_sets = []
                    for set in results[model_name][metric]:

                        aux_series_set = []
                        aux_series_set.append(
                            pd.Series(
                                [model_name] * n_folds, name="models"
                            ).reset_index(drop=True)
                        )
                        aux_series_set.append(
                            pd.Series([set] * n_folds, name="set").reset_index(
                                drop=True
                            )
                        )
                        aux_series_set.append(
                            pd.Series(results[model_name]["FIT_TIME"], name="FIT_TIME")
                            .explode()
                            .reset_index(drop=True)
                        )
                        aux_series_set.append(
                            pd.Series(
                                results[model_name][metric][set], name=f"{metric}"
                            )
                            .explode()
                            .reset_index(drop=True)
                        )
                        aux_df_sets.append(
                            pd.concat(aux_series_set, axis=1).set_index(
                                ["models", "set", "FIT_TIME"]
                            )
                        )

                    aux_df_metrics.append(pd.concat(aux_df_sets, axis=0))

            aux_df_models.append(pd.concat(aux_df_metrics, axis=1).reset_index())

    df_results_model = pd.concat(aux_df_models, axis=0)

    if not slim:
        df_results_model = df_results_model.pivot(
            index=["models", "FIT_TIME"], columns="set"
        ).reset_index()
        df_results_model.columns = [
            f"{level0}_{level1}".upper() if level0 != "" else level1
            for level1, level0 in df_results_model.columns
        ]

    return df_results_model


def RandomizedSearch(pipe,search_space:list,X_train, y_train,scoring:str=None,seed=42,n_iter=200,n_folds=5,verbose=1,n_jobs=-1, **kwargs):
    """[Makes the processes of Randomized Search through the searchspace]
    Args:
        pipe ([Pipeline]): [A object of that type is instantiated for each grid point]
        search_space (list of dicts): [List with the search space]
        X_train ([pd.DataFrame]): [The data to fit]
        Y_train ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        scoring ([str], optional): [Strategy to evaluate the performance of the cross-validated model on the test set]. Defaults to None.
        seed (int, optional): [random number generator]. Defaults to 42.
        n_iter (int, optional): [Number of parameter settings that are sampled]. Defaults to 200.
        n_folds (int, optional): [Specify the number of folds]. Defaults to 5.
        verbose (int, optional): [Controls the verbosity]. Defaults to 1.
        n_jobs (int, optional): [Number of jobs to run in parallel]. Defaults to -1.
    Returns:
        [Sklearn object]: [Returns with the object of the processes of Randomized Search]
    """
    optimizer = RandomizedSearchCV(pipe
                            , search_space
                            , cv=n_folds
                            , verbose=verbose
                            , scoring=scoring
                            , n_iter=n_iter
                            , random_state=seed
                            , return_train_score=True
                            , n_jobs=n_jobs
                            , **kwargs
                            )

    optimizer = optimizer.fit(X_train, y_train)

    return optimizer


def make_dnn_model(network_layers: Iterable[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any], activation="relu", k_initializer='glorot_normal', dropout_rate = 0):

    keras.backend.clear_session()
    gc.collect()
    model = keras.Sequential()
    inp = Input(shape=(meta["n_features_in_"]))
    model.add(inp)
    for index, layers in enumerate(network_layers):
        layer = Dense(layers, activation=activation, kernel_initializer=k_initializer, name = f'dense_{index:02d}')
        model.add(layer)
        if dropout_rate:
            model.add(AlphaDropout(dropout_rate, name = f'dropout_{index:02d}'))
            
    if meta["target_type_"] == "binary":
        n_output_units = 1
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
        
    elif meta["target_type_"] == "multiclass":
        n_output_units = meta["n_classes_"]
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    else:
        raise NotImplementedError(f"Unsupported task type: {meta['target_type_']}")
    
    out = keras.layers.Dense(n_output_units, activation=output_activation)
    model.add(out)
    model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
    return model