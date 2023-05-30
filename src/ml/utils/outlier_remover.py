import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TargetOutlierRemover(BaseEstimator, TransformerMixin):
    """Defining Outlier Remover Class ,"""

    # https://towardsdatascience.com/creating-custom-transformers-using-scikit-learn-5f9db7d7fdb5
    def __init__(self, factor=1.5, method="IQR"):
        self.factor = factor
        self.method = method

    def IQR_outlier_detector(self, X, y=None):
        """Define IQR removal"""
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound = q1 - (self.factor * iqr)
        self.upper_bound = q3 + (self.factor * iqr)

    def fit(self, X, y=None):
        if self.method == "IQR":
            self.lower_bound = []
            self.upper_bound = []
            self.IQR_outlier_detector(X)
            return self

    def transform(self, X, y=None):
        X = pd.Series(X).copy()
        if self.method == "IQR":
            X[(X < self.lower_bound) | (X > self.upper_bound)] = np.nan
            return X

    def transform_df(self, X: pd.DataFrame, y: pd.Series):
        if self.method == "IQR":
            y = self.transform(y).dropna()
            X = X.loc[y.index]
            return X, y


class FeatureOutlierRemover(BaseEstimator, TransformerMixin):
    """Defining Outlier Remover Class ,"""

    # https://towardsdatascience.com/creating-custom-transformers-using-scikit-learn-5f9db7d7fdb5
    def __init__(self, factor=1.5, method="IQR"):
        self.factor = factor
        self.method = method

    def IQR_outlier_detector(self, X, y=None):
        """Define IQR removal"""
        X = pd.Series(X).copy()
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self, X, y=None):
        if self.method == "IQR":
            self.lower_bound = []
            self.upper_bound = []
            X.apply(self.IQR_outlier_detector)
            return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        if self.method == "IQR":
            for i in range(X.shape[1]):
                x = X.iloc[:, i].copy()
                x[(x < self.lower_bound[i]) | (x > self.upper_bound[i])] = np.nan
                X.iloc[:, i] = x
        return X
