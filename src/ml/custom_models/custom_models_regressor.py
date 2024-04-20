import os.path as osp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaseSegmentEstimator(BaseEstimator, RegressorMixin):
    """Base Class gathering the models for each segment (part)"""

    def __init__(self, segment: str, base_path: str, fallback_path=None):
        """Initialize the model for the given segment

        Args:
            segment (str): The name of the column/field that segments each instances
            base_path (str): The path to the directory containing the models
            fallback_path (str, optional): The path to the directory containing the fallback model.
                                            If None no fallback strategy is implemented. Defaults to None.
        """
        self.segment = segment
        self.fallback_path = fallback_path
        self.base_path = base_path

    def load_models(self, base_path: str, segments: list):
        """Load the models for the given segments

        Args:
            base_path (str): The path to the directory containing the models
            segments (list): The segments for which to load the models
        """
        from src.aws.utils import read_joblib

        for segment in segments:
            self.__setattr__(
                f"model_{segment}",
                read_joblib(osp.join(base_path, segment, "model.joblib")),
            )
        if self.fallback_path:
            self.model_fallback = read_joblib(
                osp.join(base_path, self.fallback_path, "model.joblib")
            )

    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        # Get the segment
        self.segments = np.unique(X[self.segment]) if not segments else segments

        # load models
        self.load_models(self.base_path, self.segments)

    def preprocess_input(self, X):

        # Check if X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            if "instances" in X:
                X = pd.DataFrame(X["instances"])
            else:
                X = pd.DataFrame.from_records(X)
        return X

    def postprocess_predictions(self, y_pred):
        return {"predictions": [{"0.5": [prediction]} for prediction in y_pred.values]}

    def model_predict(self, X, segment: str):
        """Predict the target for the given segment

        Args:
            X (pd.DataFrame): The dataframe to predict the target for
            segment (str): The segment to predict the target for

        Raises:
            ValueError: If the model for the given segment is not found and ther is no fallback model

        Returns:
            np.array: The predicted target for the given segment
        """

        # Get the model for the given segment
        if segment in self.segments:
            return self.__getattribute__(f"model_{segment}").predict(X)

        # if segment is not in the models, use the fallback model
        elif self.fallback_path:
            return self.model_fallback.predict(X)
        else:
            raise ValueError(f"No model for segment {segment}")

    def predict(self, X, postprocess=True):
        """Predict the target for the given segments

        Args:
            X (list, pd.DataFrame): The dataframe or list of jsons to predict the target for
            postprocess (bool, optional): If True the predictions are postprocessed. Defaults to True.
        Returns:
            np.array: The predicted target for the given segments
        """

        # preprocess X
        X = self.preprocess_input(X)

        # initialize predictions with NaN
        y_pred = pd.Series([np.nan] * len(X), index=X.index)

        # fill predictions with predictions from each segment
        for segment in self.segments:
            if len(X[X[self.segment] == segment]) > 0:
                mask = X[self.segment] == segment
                y_pred[mask] = self.model_predict(X[mask], segment)

        return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
