import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance (Haversine) between two points.

    Args:
        lat1, lon1: Latitude and Longitude of point 1 in decimal degrees.
        lat2, lon2: Latitude and Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometers as a float.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c
