import numpy as np
import pandas as pd
from sklearn.metrics import (
    get_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def RSME(errors=None):
    """Root Mean Squared Error

    Args:
        errors (pd.Series, optional): Column with the calculated errors

    Returns:
        float: Calculated Root Mean Squared Error
    """

    return np.sqrt(np.mean((errors) ** 2))


def MAE(errors=None):
    """Mean Absolute Error

    Args:
        errors (pd.Series, optional): Column with the calculated errors

    Returns:
        float: Calculated Mean Absolute Error
    """

    return np.mean(abs(errors), axis=0)


def get_metrics_error(df, by, setup_dict, slim=True, task="regression"):
    """get metrics of error for the desired subsets (by)

    Args:
        df ([pandas dataframe]): pandas dataframe
        by ([list or str]): column names to be grouped by
        setup_dict ([list]): must be a list of dicts in this format:
                                                    setup_dict=[{'name': name of step
                                                                , 'real': column name of y_true
                                                                , 'estimated': column name of y_pred},
        slim (bool, optional): If the df should be slim (with the "name" on rows) or not. Defaults to False.
        task (str): "regression" or "classification". Default is "regression"

    Returns:
        [pandas dataframe]: pandas dataframe with calculated errors
    """
    df_groupby = (
        df.groupby(by)
        .apply(lambda df: pd.Series(_create_metrics_dict(df, setup_dict, task=task)))
        .dropna(subset=["count_orders"])
        .reset_index()
    )
    # normalize all columns
    df_list = list()
    num_steps = len(setup_dict)

    for col in df_groupby.columns[-num_steps:]:
        metrics = pd.json_normalize(df_groupby[col])
        if not slim:
            metrics.columns = [f"{col}_{c}" for c in metrics.columns]
        else:
            df_aux = df_groupby.iloc[:, :-num_steps]
            df_aux["name"] = col
            metrics = df_aux.join(metrics)

        df_list.append(metrics)

    # combine into one dataframe
    if not slim:
        return pd.concat([df_groupby.iloc[:, :-num_steps]] + df_list, axis=1)
    else:
        return pd.concat(df_list, axis=0)


def _create_metrics_dict(df, setup_dict, task):
    """[Aux function to create the metrics dict]

    Args:
        df ([pandas dataframe]): pandas dataframe
        setup_dict ([type]): must be a list of dicts in this format:
                                                    setup_dict=[{'name': name of step
                                                                , 'real': column name of y_true
                                                                , 'estimated': column name of y_pred},
                                                                , 'proba': column name of y_proba}}] *for classification* - works as `y_score` sklearn
                                                                    - Reference: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        task (str): "regression" or "classification"

    Returns:
        [dict]: [dict of metrics]
    """
    metrics_dict = {"count_orders": len(df)}
    for step in setup_dict:

        if task == "regression":
            metrics_dict[step["name"]] = {
                "real_mean": np.mean(df[step["real"]]),
                "estimated_mean": np.mean(df[step["estimated"]]),
                "error_mean": np.mean(df[step["estimated"]] - df[step["real"]]),
                "real_median": np.median(df[step["real"]]),
                "estimated_median": np.median(df[step["estimated"]]),
                "error_median": np.median(df[step["estimated"]] - df[step["real"]]),
                "RSME": mean_squared_error(
                    df[step["real"]], df[step["estimated"]], squared=False
                ),
                "MAE": mean_absolute_error(df[step["real"]], df[step["estimated"]]),
                "MAPE": mean_absolute_percentage_error(
                    df[step["real"]], df[step["estimated"]]
                ),
                "R^2": r2_score(df[step["real"]], df[step["estimated"]])
                # 'on_time':np.mean(df[step['on_time']]),
                # 'late':np.mean(df[step['late']]),
                # 'early':np.mean(df[step['early']])
            }
        elif task == "classification":
            metrics_dict[step["name"]] = {
                "accuracy": get_metric_result(
                    "accuracy", y_true=df[step["real"]], y_pred=df[step["estimated"]]
                ),
                "balanced_acc": get_metric_result(
                    "balanced_accuracy",
                    y_true=df[step["real"]],
                    y_pred=df[step["estimated"]],
                ),
                "F1": get_metric_result(
                    "f1", y_true=df[step["real"]], y_pred=df[step["estimated"]]
                ),
                "Precision": get_metric_result(
                    "precision", y_true=df[step["real"]], y_pred=df[step["estimated"]]
                ),
                "recall": get_metric_result(
                    "recall", y_true=df[step["real"]], y_pred=df[step["estimated"]]
                ),
                "roc_auc": get_metric_result(
                    "roc_auc", y_true=df[step["real"]], y_proba=df[step["proba"]]
                ),
            }
        else:
            raise AttributeError(f'The task "{task}" is not a valid task')

    return metrics_dict


def get_metric_result(metric, y_true, y_pred=None, y_proba=None):
    """_summary_

    Args:
        metric (str or callable): Scoring method as string. If callable it is returned as is.
        y_true (1d array-like, or label indicator array / sparse matrix): Ground truth (correct) labels.
        y_pred (1d array-like, or label indicator array / sparse matrix): Predicted labels, as returned by a model.
        y_proba (1d array-like, optional): Probabilities associated with each class.

    Returns:
        float: metric result
    """
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    if y_proba is None:
        pass
    elif y_proba.ndim == 2:
        y_proba = y_proba.reshape(-1) if y_proba.shape[1] == 1 else y_proba[:, 1]

    scorer = get_scorer(metric)
    if any(
        [
            arg in scorer._factory_args()
            for arg in ("needs_threshold=True", "needs_proba=True")
        ]
    ):
        return scorer._sign * scorer._score_func(y_true, y_proba, **scorer._kwargs)
    else:
        return scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)
