import os.path as osp
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseSegmentClassifier(BaseEstimator, ClassifierMixin):
    """Base Class gathering the models for each segment for classification tasks"""

    def __init__(
        self,
        segment: str,
        base_path: str,
        fallback_path=None,
        model_name="model.p",
    ):
        """Initialize the model for the given segment for classification"""
        self.segment = segment
        self.fallback_path = fallback_path
        self.base_path = base_path
        self.model_name = model_name
        # Initialize classes_ to None; will be set in fit
        self.classes_ = None

    def load_models(self, base_path: str, segments: list):
        """Load the models for the given segments"""
        from src.ml.utils.utils import (
            read_joblib,
        )

        for segment in segments:
            self.__setattr__(
                f"model_{segment}",
                read_joblib(osp.join(base_path, segment, self.model_name)),
            )
            model_ = self.__getattribute__(f"model_{segment}")
            # setting classes_ from the first model loaded
            if self.classes_ is None and hasattr(model_, "classes_"):
                self.classes_ = model_.classes_
        if self.fallback_path:
            self.model_fallback = read_joblib(
                osp.join(base_path, self.fallback_path, self.model_name)
            )

    def fit(self, X=None, segments=None):
        """Models are pre-trained and loaded, so 'fit' only prepares model loading."""
        self.segments = np.unique(X[self.segment]) if not segments else segments
        self.load_models(self.base_path, self.segments)

    def preprocess_input(self, X):
        """Preprocess input data"""
        if not isinstance(X, pd.DataFrame):
            if "instances" in X:
                X = pd.DataFrame(X["instances"])
            else:
                X = pd.DataFrame.from_records(X)
        return X

    def model_predict(self, X, segment: str):
        """Predict the class labels for the given segment"""
        if segment in self.segments:
            return self.__getattribute__(f"model_{segment}").predict(X)
        elif self.fallback_path:
            return self.model_fallback.predict(X)
        else:
            raise ValueError(f"No model for segment {segment}")

    def predict(self, X):
        """Predict class labels for the given input"""
        X = self.preprocess_input(X)
        y_pred = pd.Series([None] * len(X), index=X.index)

        for segment in self.segments:
            mask = X[self.segment] == segment
            if mask.any():
                y_pred[mask] = self.model_predict(X[mask], segment)

        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities for the given input"""
        X = self.preprocess_input(X)
        # Initialize a DataFrame to hold the probability arrays, assuming a known number of classes.
        # This example assumes that 'self.classes_' is defined and contains the list of class labels.
        num_classes = len(self.classes_)
        y_proba = pd.DataFrame(
            np.nan, index=X.index, columns=[f"Class_{i}" for i in range(num_classes)]
        )

        for segment in self.segments:
            mask = X[self.segment] == segment
            if mask.any():
                model = self.__getattribute__(f"model_{segment}")
                if hasattr(model, "predict_proba"):
                    # Assuming predict_proba returns a 2D array with shape (n_samples, n_classes)
                    proba = model.predict_proba(X[mask])
                    y_proba.loc[mask, :] = proba
                else:
                    # Handle models without predict_proba, e.g., by assigning uniform probabilities
                    uniform_proba = np.ones((sum(mask), num_classes)) / num_classes
                    y_proba.loc[mask, :] = uniform_proba

        return y_proba.values
