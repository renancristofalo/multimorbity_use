import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .outlier_remover import FeatureOutlierRemover


class BypassTransformer(TransformerMixin):
    """Define a custom transformer that can be used as a bypass preprocessing step in a pipeline.
    This transformer does not do any preprocessing, but can be used to bypass preprocessing"""

    def __init__(self, **kwargs):
        self.hyperparam = kwargs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X  # no transformation


def select_preprocessor(preprocessor, X, bypass_columns=[]):
    """Selects the preprocessor to use.

    Args:
        preprocessor (str, ColumnTransformer): The preprocessor to use.
        X (_type_): The data used in the preprocessor fit.
        bypass_columns (list, optional): Columns to bypass the preprocessor.

    Raises:
        NotImplementedError: If the preprocessor string is not implemented.

    Returns:
        ColumnTransformer: defined preprocessor
    """

    X = X[[col for col in X.columns if col not in bypass_columns]]

    if preprocessor == "default":
        float_features = [
            col
            for col in X.select_dtypes(["float16", "float32", "float64"]).columns
            if col not in bypass_columns
        ]
        int_features = [
            col
            for col in X.select_dtypes(["int16", "int32", "int64"]).columns
            if col not in bypass_columns
        ]
        categorical_features_low_granularity = [
            col
            for col in X.select_dtypes([object, "category"]).columns
            if X[col].nunique() < 20 and col not in bypass_columns
        ]
        categorical_features_high_granularity = [
            col
            for col in X.select_dtypes([object, "category"]).columns
            if col not in categorical_features_low_granularity
            and col not in bypass_columns
        ]

        # For other methods of inputation: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
        float_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        int_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]
        )

        # Applying SimpleImputer and then OneHotEncoder
        categorical_transformer_low_granularity = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
            ]
        )

        # Applying SimpleImputer and then OneHotEncoder
        categorical_transformer_high_granularity = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                ),
                ("imputer_unknown", SimpleImputer(strategy="most_frequent")),
            ]
        )

        bypass_transformer = Pipeline(steps=[("Bypass", BypassTransformer())])

        # Wrap all the steps onto a single Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("float", float_transformer, float_features),
                ("int", int_transformer, int_features),
                (
                    "categorical_low_granularity",
                    categorical_transformer_low_granularity,
                    categorical_features_low_granularity,
                ),
                (
                    "categorical_high_granularity",
                    categorical_transformer_high_granularity,
                    categorical_features_high_granularity,
                ),
                ("bypass", bypass_transformer, bypass_columns),
            ],
            remainder="passthrough",
        )

    elif preprocessor == "TargetEncoding":
        float_features = [
            col
            for col in X.select_dtypes(["float16", "float32", "float64"]).columns
            if col not in bypass_columns
        ]
        int_features = [
            col
            for col in X.select_dtypes(["int16", "int32", "int64"]).columns
            if col not in bypass_columns
        ]
        categorical_features_low_granularity = [
            col
            for col in X.select_dtypes([object, "category"]).columns
            if X[col].nunique() < 20 and col not in bypass_columns
        ]
        categorical_features_high_granularity = [
            col
            for col in X.select_dtypes([object, "category"]).columns
            if col not in categorical_features_low_granularity
            and col not in bypass_columns
        ]

        # For other methods of inputation: https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html
        float_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        int_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]
        )

        # Applying SimpleImputer and then OneHotEncoder
        categorical_transformer_low_granularity = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
            ]
        )

        # Applying SimpleImputer and then OneHotEncoder
        categorical_transformer_high_granularity = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", TargetEncoder(handle_unknown="value"),),
            ]
        )

        bypass_transformer = Pipeline(steps=[("Bypass", BypassTransformer())])

        # Wrap all the steps onto a single Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("float", float_transformer, float_features),
                ("int", int_transformer, int_features),
                (
                    "categorical_low_granularity",
                    categorical_transformer_low_granularity,
                    categorical_features_low_granularity,
                ),
                (
                    "categorical_high_granularity",
                    categorical_transformer_high_granularity,
                    categorical_features_high_granularity,
                ),
                ("bypass", bypass_transformer, bypass_columns),
            ],
            remainder="passthrough",
        )

    elif isinstance(preprocessor, str):
        raise NotImplementedError(f"The preprocessor {preprocessor} is not implemented")

    return preprocessor
