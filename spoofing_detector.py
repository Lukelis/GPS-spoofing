import pandas as pd
from utils import haversine


def detect_spoofing(vessel_df, speed_threshold=100, jump_distance_km=50, jump_time_sec=300):
    """
    Detect GPS spoofing anomalies in a single vessel's AIS data.

    Args:
        vessel_df (pd.DataFrame): AIS data for a single vessel.
        speed_threshold (float): Speed in km/h considered unrealistic.
        jump_distance_km (float): Distance jump considered anomalous.
        jump_time_sec (int): Time window for distance jumps.

    Returns:
        pd.DataFrame: Rows with detected anomalies.
    """
    if len(vessel_df) < 2:
        return None

    vessel_df = vessel_df.sort_values("timestamp").reset_index(drop=True)

    # Shifted values
    vessel_df["lat_prev"] = vessel_df["Latitude"].shift()
    vessel_df["lon_prev"] = vessel_df["Longitude"].shift()
    vessel_df["time_prev"] = vessel_df["timestamp"].shift()
    vessel_df.dropna(subset=["lat_prev", "lon_prev", "time_prev"], inplace=True)

    # Calculate distance and speed
    vessel_df["dist_km"] = haversine(vessel_df["lat_prev"], vessel_df["lon_prev"],
                                      vessel_df["Latitude"], vessel_df["Longitude"])
    vessel_df["time_diff_sec"] = (vessel_df["timestamp"] - vessel_df["time_prev"]).dt.total_seconds()
    vessel_df["speed_kmph"] = vessel_df["dist_km"] / (vessel_df["time_diff_sec"] / 3600)

    # Flag anomalies
    flags = (vessel_df["speed_kmph"] > speed_threshold) | \
            ((vessel_df["dist_km"] > jump_distance_km) & (vessel_df["time_diff_sec"] < jump_time_sec))

    anomalies = vessel_df[flags].copy()
    return anomalies[["MMSI", "timestamp", "Latitude", "Longitude", "speed_kmph", "dist_km"]]
