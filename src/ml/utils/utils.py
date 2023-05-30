import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.pipeline import Pipeline


def get_feature_out(estimator, feature_in):
    if hasattr(estimator, "get_feature_names"):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f"vec___-{f}" for f in estimator.get_feature_names_out()]
        else:
            features_out = []
            for feature, category, drop_idx in zip(
                feature_in, estimator.categories_, estimator.drop_idx_
            ):
                if drop_idx is not None:
                    category = np.delete(category, drop_idx)
                features_out.extend(
                    [f"{feature}___-{category_}" for category_ in category]
                )
            return features_out
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder !='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if features:
            if name != "remainder":
                if isinstance(estimator, Pipeline):
                    current_features = features
                    for step in estimator:
                        current_features = get_feature_out(step, current_features)

                    features_out = current_features
                else:
                    features_out = get_feature_out(estimator, features)
                output_features.extend(features_out)
            elif estimator == "passthrough":
                output_features.extend(ct.feature_names_in_[features])

    return output_features


def write_joblib(value, path):
    """Function to write a joblib file to an s3 bucket or local directory.

    Args:
        value: The value that you want to save
        path (str): an s3 bucket or local directory path.
        kwargs: additional arguments to be passed to `s3fs`

    """
    # Path is a local directory

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        joblib.dump(value, f)


def read_joblib(path):
    """Function to load a joblib file from an s3 bucket or local directory.

    Args:
        path (str): an s3 bucket or local directory path where the file is stored
        kwargs: additional arguments to be passed to `s3fs`

    Returns:
        file: Joblib file loaded
    """

    # Path is a local directory
    with open(path, "rb") as f:
        file = joblib.load(f)

    return file
