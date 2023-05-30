import haversine as hs


def get_haversine_distance(lon_1, lat_1, lon_2, lat_2):
    """[Calculates the haversine distance between to points in the world, in meters]

    Args:
        lon_1 ([float]): [Longitude of point 1]
        lat_1 ([float]): [Latitude of point 1]
        lon_2 ([float]): [Longitude of point 2]
        lat_2 ([float]): [Latitude of point 2]

    Returns:
        [type]: [haversine distance]
    """
    try:
        distance = hs.haversine((lon_1, lat_1), (lon_2, lat_2), unit=hs.Unit.METERS)
    except:
        distance = None
    return distance
