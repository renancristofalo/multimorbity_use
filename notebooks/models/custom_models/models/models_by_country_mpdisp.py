import numpy as np
from src.ml.custom_models.custom_models import BaseSegmentEstimator
from src.ml.data_preparation.data_preparation import get_haversine_distance


class TotalAssigningModelByCountryMPDisp(BaseSegmentEstimator):
    def __init__(self):
        super().__init__(
            segment="store_branch_country_id",
            base_path="models/multipicking_dispatcher/real_total_assigning/models",
            fallback_path="global",
        )

    def calculate_haversine_distance(self, X, lat_1, lon_1, lat_2, lon_2):
        return np.vectorize(get_haversine_distance)(
            X[lat_1], X[lon_1], X[lat_2], X[lon_2], unit="km"
        )

    def predict(self, X, postprocess=True):
        X = self.preprocess_input(X)
        X["origin_destination_haversine_distance"] = self.calculate_haversine_distance(
            X, "origin_lat", "origin_lng", "destination_lat", "destination_lng"
        )
        return super().predict(X=X, postprocess=postprocess)


class ShoppingSetupModelByCountryMPDisp(BaseSegmentEstimator):
    def __init__(self):
        super().__init__(
            segment="store_branch_country_id",
            base_path="models/multipicking_dispatcher/real_shopping_setup/models",
            fallback_path="global",
        )

    def calculate_haversine_distance(self, X, lat_1, lon_1, lat_2, lon_2):
        return np.vectorize(get_haversine_distance)(
            X[lat_1], X[lon_1], X[lat_2], X[lon_2], unit="km"
        )

    def predict(self, X, postprocess=True):
        X = self.preprocess_input(X)
        X["origin_destination_haversine_distance"] = self.calculate_haversine_distance(
            X, "origin_lat", "origin_lng", "destination_lat", "destination_lng"
        )
        return super().predict(X=X, postprocess=postprocess)


class ShoppingModelByCountryMPDisp(BaseSegmentEstimator):
    def __init__(self):
        super().__init__(
            segment="store_branch_country_id",
            base_path="models/multipicking_dispatcher/real_shopping/models",
            fallback_path="global",
        )

    def calculate_haversine_distance(self, X, lat_1, lon_1, lat_2, lon_2):
        return np.vectorize(get_haversine_distance)(
            X[lat_1], X[lon_1], X[lat_2], X[lon_2], unit="km"
        )

    def predict(self, X, postprocess=True):
        X = self.preprocess_input(X)
        X["origin_destination_haversine_distance"] = self.calculate_haversine_distance(
            X, "origin_lat", "origin_lng", "destination_lat", "destination_lng"
        )
        return super().predict(X=X, postprocess=postprocess)


class TransportationSetupModelByCountryMPDisp(BaseSegmentEstimator):
    def __init__(self):
        super().__init__(
            segment="store_branch_country_id",
            base_path="models/multipicking_dispatcher/real_transportation_setup/models",
            fallback_path="global",
        )

    def calculate_haversine_distance(self, X, lat_1, lon_1, lat_2, lon_2):
        return np.vectorize(get_haversine_distance)(
            X[lat_1], X[lon_1], X[lat_2], X[lon_2], unit="km"
        )

    def predict(self, X, postprocess=True):
        X = self.preprocess_input(X)
        X["origin_destination_haversine_distance"] = self.calculate_haversine_distance(
            X, "origin_lat", "origin_lng", "destination_lat", "destination_lng"
        )
        return super().predict(X=X, postprocess=postprocess)


class TransportationModelByCountryMPDisp(BaseSegmentEstimator):
    def __init__(self):
        super().__init__(
            segment="store_branch_country_id",
            base_path="models/multipicking_dispatcher/real_transportation/models",
            fallback_path="global",
        )

    def calculate_haversine_distance(self, X, lat_1, lon_1, lat_2, lon_2):
        return np.vectorize(get_haversine_distance)(
            X[lat_1], X[lon_1], X[lat_2], X[lon_2], unit="km"
        )

    def predict(self, X, postprocess=True):
        X = self.preprocess_input(X)
        X["origin_destination_haversine_distance"] = self.calculate_haversine_distance(
            X, "origin_lat", "origin_lng", "destination_lat", "destination_lng"
        )
        return super().predict(X=X, postprocess=postprocess)
