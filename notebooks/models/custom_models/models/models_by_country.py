import numpy as np
import pandas as pd
from src.ml.custom_models.custom_models import BaseSegmentEstimator

from models.models_by_country_mpdisp import (
    ShoppingModelByCountryMPDisp, ShoppingSetupModelByCountryMPDisp,
    TotalAssigningModelByCountryMPDisp, TransportationModelByCountryMPDisp,
    TransportationSetupModelByCountryMPDisp)
from models.models_by_country_single import (
    ShoppingModelByCountrySingle, ShoppingSetupModelByCountrySingle,
    TotalAssigningModelByCountrySingle, TransportationModelByCountrySingle,
    TransportationSetupModelByCountrySingle)


class TotalAssigningModelByCountry(BaseSegmentEstimator):
    def __init__(self, multipicking:str = None, segments = (0, 1)):
        """Initialize the wrap that points for mp and single orders

        Args:
            multipicking (str, optional): Multipicking col/input that is gonna separate models. Defaults to None.
            segments (list, optional): Segments for multipicking representing (False, True). Defaults to (0, 1).
        """       
        self.model_single = TotalAssigningModelByCountrySingle()
        self.model_mpdisp = TotalAssigningModelByCountryMPDisp()
        self.multipicking = multipicking
        self.segments = segments
        
    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        
        self.model_single.fit(X=X, segments=segments)
        self.model_mpdisp.fit(X=X, segments=segments)
    
    def predict(self, X, postprocess=True):
              
        X = self.preprocess_input(X)
        
        if self.multipicking in X.columns:
            
            # treating null:
            if X[self.multipicking].isnull().sum():
               X[self.multipicking].fillna(self.segments[0],inplace = True) 
               
            # initialize predictions with NaN
            y_pred = pd.Series(np.nan, index=X.index)
            
            # fill predictions with predictions from each segment
            for segment in self.segments:
                if len(X[X[self.multipicking] == segment]) > 0:
                    mask = X[self.multipicking] == segment
                    if segment > 0:
                        y_pred[mask] = self.model_mpdisp.predict(X[mask], postprocess = False)
                    else:
                        y_pred[mask] = self.model_single.predict(X[mask], postprocess = False)
                        
            return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
        else:
            return self.model_single.predict(X, postprocess = postprocess)


class ShoppingSetupModelByCountry(BaseSegmentEstimator):
    def __init__(self, multipicking:str = None, segments = (0, 1)):
        """Initialize the wrap that points for mp and single orders

        Args:
            multipicking (str, optional): Multipicking col/input that is gonna separate models. Defaults to None.
            segments (list, optional): Segments for multipicking representing (False, True). Defaults to (0, 1).
        """       
        self.model_single = ShoppingSetupModelByCountrySingle()
        self.model_mpdisp = ShoppingSetupModelByCountryMPDisp()
        self.multipicking = multipicking
        self.segments = segments
        
    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        
        self.model_single.fit(X=X, segments=segments)
        self.model_mpdisp.fit(X=X, segments=segments)
    
    def predict(self, X, postprocess=True):
              
        X = self.preprocess_input(X)
        
        if self.multipicking in X.columns:
            
            # treating null:
            if X[self.multipicking].isnull().sum():
               X[self.multipicking].fillna(self.segments[0],inplace = True) 
               
            # initialize predictions with NaN
            y_pred = pd.Series(np.nan, index=X.index)
            
            # fill predictions with predictions from each segment
            for segment in self.segments:
                if len(X[X[self.multipicking] == segment]) > 0:
                    mask = X[self.multipicking] == segment
                    if segment > 0:
                        y_pred[mask] = self.model_mpdisp.predict(X[mask], postprocess = False)
                    else:
                        y_pred[mask] = self.model_single.predict(X[mask], postprocess = False)
                        
            return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
        else:
            return self.model_single.predict(X, postprocess = postprocess)



class ShoppingModelByCountry(BaseSegmentEstimator):
    def __init__(self, multipicking:str = None, segments = (0, 1)):
        """Initialize the wrap that points for mp and single orders

        Args:
            multipicking (str, optional): Multipicking col/input that is gonna separate models. Defaults to None.
            segments (list, optional): Segments for multipicking representing (False, True). Defaults to (0, 1).
        """       
        self.model_single = ShoppingModelByCountrySingle()
        self.model_mpdisp = ShoppingModelByCountryMPDisp()
        self.multipicking = multipicking
        self.segments = segments
        
    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        
        self.model_single.fit(X=X, segments=segments)
        self.model_mpdisp.fit(X=X, segments=segments)
    
    def predict(self, X, postprocess=True):
              
        X = self.preprocess_input(X)
        
        if self.multipicking in X.columns:
            
            # treating null:
            if X[self.multipicking].isnull().sum():
               X[self.multipicking].fillna(self.segments[0],inplace = True) 
               
            # initialize predictions with NaN
            y_pred = pd.Series(np.nan, index=X.index)
            
            # fill predictions with predictions from each segment
            for segment in self.segments:
                if len(X[X[self.multipicking] == segment]) > 0:
                    mask = X[self.multipicking] == segment
                    if segment > 0:
                        y_pred[mask] = self.model_mpdisp.predict(X[mask], postprocess = False)
                    else:
                        y_pred[mask] = self.model_single.predict(X[mask], postprocess = False)
                        
            return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
        else:
            return self.model_single.predict(X, postprocess = postprocess)

class TransportationSetupModelByCountry(BaseSegmentEstimator):
    def __init__(self, multipicking:str = None, segments = (0, 1)):
        """Initialize the wrap that points for mp and single orders

        Args:
            multipicking (str, optional): Multipicking col/input that is gonna separate models. Defaults to None.
            segments (list, optional): Segments for multipicking representing (False, True). Defaults to (0, 1).
        """       
        self.model_single = TransportationSetupModelByCountrySingle()
        self.model_mpdisp = TransportationSetupModelByCountryMPDisp()
        self.multipicking = multipicking
        self.segments = segments
        
    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        
        self.model_single.fit(X=X, segments=segments)
        self.model_mpdisp.fit(X=X, segments=segments)
    
    def predict(self, X, postprocess=True):
              
        X = self.preprocess_input(X)
        
        if self.multipicking in X.columns:
            
            # treating null:
            if X[self.multipicking].isnull().sum():
               X[self.multipicking].fillna(self.segments[0],inplace = True) 
               
            # initialize predictions with NaN
            y_pred = pd.Series(np.nan, index=X.index)
            
            # fill predictions with predictions from each segment
            for segment in self.segments:
                if len(X[X[self.multipicking] == segment]) > 0:
                    mask = X[self.multipicking] == segment
                    if segment > 0:
                        y_pred[mask] = self.model_mpdisp.predict(X[mask], postprocess = False)
                    else:
                        y_pred[mask] = self.model_single.predict(X[mask], postprocess = False)
                        
            return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
        else:
            return self.model_single.predict(X, postprocess = postprocess)


class TransportationModelByCountry(BaseSegmentEstimator):
    def __init__(self, multipicking:str = None, segments = (0, 1)):
        """Initialize the wrap that points for mp and single orders

        Args:
            multipicking (str, optional): Multipicking col/input that is gonna separate models. Defaults to None.
            segments (list, optional): Segments for multipicking representing (False, True). Defaults to (0, 1).
        """       
        self.model_single = TransportationModelByCountrySingle()
        self.model_mpdisp = TransportationModelByCountryMPDisp()
        self.multipicking = multipicking
        self.segments = segments
        
    def fit(self, X=None, segments=None):
        """Read the models for the given segments. If no segments are given, all models are loaded based on the X DataFrame.

        Args:
            X (pd.DataFrame, optional): Dataframe to gather the unique segments. Defaults to None.
            segments (_type_, optional): List of segments to use. Defaults to None.
        """
        
        self.model_single.fit(X=X, segments=segments)
        self.model_mpdisp.fit(X=X, segments=segments)
    
    def predict(self, X, postprocess=True):
              
        X = self.preprocess_input(X)
        
        if self.multipicking in X.columns:
            
            # treating null:
            if X[self.multipicking].isnull().sum():
               X[self.multipicking].fillna(self.segments[0],inplace = True) 

            # initialize predictions with NaN
            y_pred = pd.Series(np.nan, index=X.index)
            
            # fill predictions with predictions from each segment
            for segment in self.segments:
                if len(X[X[self.multipicking] == segment]) > 0:
                    mask = X[self.multipicking] == segment
                    if segment > 0:
                        y_pred[mask] = self.model_mpdisp.predict(X[mask], postprocess = False)
                    else:
                        y_pred[mask] = self.model_single.predict(X[mask], postprocess = False)
                        
            return self.postprocess_predictions(y_pred) if postprocess else y_pred.values
        else:
            return self.model_single.predict(X, postprocess = postprocess)
