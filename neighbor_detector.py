import pandas as pd
from tqdm import tqdm

def detect_neighbor_conflicts(df, time_window='1min', grid_size=0.001):
    """
    Detects overlapping GPS positions among different vessels at similar times.

    Args:
        df (pd.DataFrame): Full AIS dataset
        time_window (str): Time rounding frequency, e.g. '1min' or '5min'
        grid_size (float): Grid size for lat/lon rounding (0.001 ≈ ~100 meters)

    Returns:
        pd.DataFrame: Suspected GPS spoofing conflicts between nearby vessels
    """
    # Create rounded time and location columns
    df = df.copy()
    df["timestamp_rounded"] = df["timestamp"].dt.round(time_window)
    df["lat_rounded"] = df["Latitude"].round(3)
    df["lon_rounded"] = df["Longitude"].round(3)

    # Group by time and position grid
    grouped = df.groupby(["timestamp_rounded", "lat_rounded", "lon_rounded"])



    # Filter groups with more than 1 unique vessel
    conflict_rows = []
    for _, group in tqdm(grouped, desc="Scanning for neighbor conflicts"):
        if group["MMSI"].nunique() > 1:
            conflict_rows.append(group)

    if conflict_rows:
        conflicts_df = pd.concat(conflict_rows)
        return conflicts_df.reset_index(drop=True)
    else:
        return pd.DataFrame()
