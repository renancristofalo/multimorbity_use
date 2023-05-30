import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from BorutaShap import BorutaShap
from sklearn.base import clone, is_classifier
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       f_classif, r_regression)
from sklearn.linear_model import LassoCV
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.ml.evaluation.evaluate import get_metric_result, get_scorer
from src.utils.utils import now_str, time_diff_str


def collinear_removal(X, y, correlation_threshold):
    """
    Finds collinear features based on the correlation coefficient between features.
    For each pair of features with a correlation coefficient greather than `correlation_threshold`,
    only one of the pair is identified for removal.

    Using code adapted from: https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

    Parameters
    --------

    correlation_threshold : float between 0 and 1
        Value of the Pearson correlation cofficient for identifying correlation features

    one_hot : boolean, default = False
        Whether to one-hot encode the features before calculating the correlation coefficients

    """

    # Calculate the correlations between every column

    y_target = pd.Series(y.copy().values, name="target")

    df = pd.concat([X, y_target], axis=1)

    corr_matrix = df.corr()

    corr_matrix = (
        corr_matrix.sort_values("target", key=lambda x: abs(x), ascending=False)
        .sort_values("target", key=lambda x: abs(x), ascending=False, axis=1)
        .drop("target", axis=1)
        .drop("target", axis=0)
    )

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [
        column
        for column in upper.columns
        if any(upper[column].abs() > correlation_threshold)
    ]

    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(
        columns=["drop_feature", "corr_feature", "corr_value"]
    )

    # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:

        # Find the correlated features
        corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

        # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]

        # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict(
            {
                "drop_feature": drop_features,
                "corr_feature": corr_features,
                "corr_value": corr_values,
            }
        )

        # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index=True)

    collinear_support = np.array(
        [True if col not in to_drop else False for col in X.columns]
    )
    collinear_features = X.columns.drop(to_drop)

    print(
        "%d features with a correlation magnitude greater than %0.2f.\n"
        % (len(to_drop), correlation_threshold)
    )

    return collinear_support, collinear_features, record_collinear


# Definição de função auxiliar para determinação da melhor quantidade de features
def loop_k_features(
    X,
    y,
    model,
    method: str = "FromModel",
    max_features=150,
    metric=None,
    scoring_function=None,
):
    """Loop through the qty of features to select which qty provides the best result using the Feature Selection models to select the best features

        Args:
            model ([sklearn model]): [Model to be used in the selection and/or for the analysis of the results]
            method (str, optional): [Method to choose the features]. Defaults to 'FromModel'.
            max_features (int, optional): [Max number of features to be tested]. Defaults to 150.
            flag_removal (bool, optional): [If this method will flag removal or not]. Defaults to True.
            metric ([str], optional): [Metric to be used in the analysis]. Defaults to None."""

    start = datetime.datetime.now()

    task = "regression" if not is_classifier(model) else "classification"

    if metric is None:
        metric = "r2" if task == "regression" else "accuracy"

    feature_names = X.columns

    # Convert to np array
    features = np.array(X)
    labels = np.array(y).reshape((-1,))

    n_features = min(len(feature_names), max_features)
    # no of features
    n_list = np.arange(1, n_features + 1)

    y_proba = None

    # Variable to store the optimum features
    nof = 0
    score_list = []

    # Creating Validation dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=123
    )

    for n in tqdm(range(len(n_list))):

        selector = _get_selector(
            method=method,
            n_list=n_list,
            n=n,
            model=model,
            scoring_function=scoring_function,
        )

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_select = selector.transform(X_test)

        clone_model = clone(model)
        clone_model.fit(X_train_selected, y_train)

        y_pred = clone_model.predict(X_test_select)
        if task == "classification":
            y_proba = clone_model.predict_proba(X_test_select)

        scorer = get_scorer(metric)
        score = get_metric_result(metric, y_test, y_pred, y_proba)

        if n == 0:
            high_score = score

        score_list.append(score)

        if score * scorer._sign >= high_score:
            high_score = score
            nof = n_list[n]
            support = selector.get_support()
            best_selector = selector
            best_model = clone_model
            selected_cols = feature_names[support]

    print("Optimum number of features: %d" % nof)
    if is_classifier(model):
        print(f"Score ({metric}) with %d features: %f" % (nof, high_score * 100))
    else:
        print("Score (R²) with %d features: %f" % (nof, high_score * 100))
    print(f"[{now_str()}] TE: {time_diff_str(start, datetime.datetime.now())}")

    # plot
    f, ax = plt.subplots(figsize=(20, 5))
    sns.lineplot(x = n_list, y = score_list, ax=ax)

    return support, features


def _get_selector(method: str, n_list, n: int, model, scoring_function=None):
    """[Get selector based on the method]

    Args:
        method (str): [String with the method selected]
        n_list ([list or array]): [number list]
        n (int): [number of features]
        model ([type]): [Model to be tested the selection]

    Returns:
        [type]: [Returns with the selector]
    """

    task = "regression" if not is_classifier(model) else "classification"
    if method == "RFE":
        return RFE(model, n_features_to_select=n_list[n])

    elif method == "KBest":
        if not scoring_function:
            scoring_function = f_classif if task == "classification" else r_regression
        return SelectKBest(score_func=scoring_function, k=n_list[n])

    elif method == "FromModel":
        return SelectFromModel(model, threshold=0, max_features=n_list[n])

    else:
        raise NotImplementedError(f"Method {method} is no implemented")


def Lasso_selection(X, y, **kwargs):
    """[Feature Selection based on the Lasso Regression]

    Args:
        X ([pd.DataFrame]): [The data to fit]
        y ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        feature_names ([List or array]): [List with all columnames]

    Returns:
        [type]: [Return with the support and the features names]
    """
    feature_names = X.columns

    # Seleção de Variáveis por meio da Lasso Regression
    reg = LassoCV(**kwargs)
    reg.fit(X, y)

    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))

    coef = pd.Series(reg.coef_, index=feature_names)
    Lasso_support = coef != 0
    Lasso_features = coef[coef != 0].index
    print(
        "Lasso picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )

    imp_coef = coef.sort_values()

    fig = plt.figure(figsize=(20, 0.15 * len(feature_names)))
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")

    return Lasso_support, Lasso_features


def BorutaShap_selection(
    X,
    y,
    model,
    kwargs={"n_trials": 100, "random_state": 0, "normalize": False, "verbose": False,},
):
    """Runs the boruta method of Feature Selection using the shapley values

    REFERENCES:
        - Explanation of the Boruta method: https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a
        - BorutaShap explanation and comparison: https://medium.com/analytics-vidhya/is-this-the-best-feature-selection-algorithm-borutashap-8bc238aa1677
        - BorutaShap: https://pypi.org/project/BorutaShap/

    Args:
        model ([sklearn model]): [Model to be used in the Boruta Method]
        flag_removal (bool, optional): [If this method will flag removal or not]. Defaults to True.
        kwargs (dict, optional): [Optional arguments for the selector]. Defaults to {'n_trials': 100, 'random_state': 0, 'normalize': False, 'verbose': False}.
    """
    task = "regression" if not is_classifier(model) else "classification"

    feature_names = X.columns

    # Convert to np array
    features = X.copy()
    labels = np.array(y).reshape((-1,))

    print("Starting BorutaShap\n")

    selector = BorutaShap(
        model=model, importance_measure="shap", classification=task == "classification",
    )

    selector.fit(X=features, y=labels, **kwargs)
    selector.TentativeRoughFix()  # force the features to be accepted or rejected

    feature_support = [feature in selector.accepted for feature in feature_names]

    print(
        f"BorutaShap picked {len(selector.accepted)} variables and eliminated the other {len(selector.rejected)} variables"
    )

    return feature_support, selector.accepted


def voting(volting_dict: dict, threshold: int, return_all_variables=False):
    """[Process of voting based on the other feature selecting methods]

    Args:
        volting_dict (dict): [Dict with names and suports for each method]
        threshold (int): [threshold to select the features]
        return_all_variables (bool, optional): [feature_selection_df should return all variables or just the selected ones]. Defaults to False.

    Returns:
        [tuple]: [Return with a list of the selected variables and a DataFrame for easier visualisation]
    """
    # Processo de Votação para escolha das melhores variáveis

    feature_selection_df = pd.DataFrame(volting_dict)

    # count
    feature_selection_df["Total"] = np.sum(feature_selection_df, axis=1)

    # print
    feature_selection_df = feature_selection_df.sort_values(
        ["Total", "Features"], ascending=False
    )
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    variáveis_selecionadas = feature_selection_df[
        feature_selection_df["Total"] >= threshold
    ]["Features"].to_list()
    # feature_selection_df
    if not return_all_variables:
        feature_selection_df = feature_selection_df[
            feature_selection_df["Total"] >= threshold
        ]

    return variáveis_selecionadas, feature_selection_df

