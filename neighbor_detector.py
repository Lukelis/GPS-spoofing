import pandas as pd
from tqdm import tqdm

def detect_neighbor_conflicts(df, 
                              time_window='2min', 
                              grid_size=0.002, 
                              max_vessels=15, 
                              mode="light"):
    """
    Detect GPS spoofing based on nearby vessels sharing time/location.

    Args:
        df (pd.DataFrame): AIS dataset
        time_window (str): e.g., '2min'
        grid_size (float): rounding for lat/lon
        max_vessels (int): max vessels allowed in a cell
        mode (str): "full", "light", "mmsi-only"

    Returns:
        pd.DataFrame or set: Conflict data or MMSI set
    """
    df = df.copy()
    df["timestamp_rounded"] = df["timestamp"].dt.round(time_window)
    df["lat_rounded"] = df["Latitude"].round(3)
    df["lon_rounded"] = df["Longitude"].round(3)

    grouped = df.groupby(["timestamp_rounded", "lat_rounded", "lon_rounded"])
    
    conflict_rows = []
    mmsi_set = set()
    included_groups = 0
    skipped_groups = 0

    for _, group in tqdm(grouped, desc="Scanning for neighbor conflicts"):
        vessel_count = group["MMSI"].nunique()
        if 1 < vessel_count <= max_vessels:
            included_groups += 1

            if mode == "mmsi-only":
                mmsi_set.update(group["MMSI"].unique())
            elif mode == "light":
                # Fix: Make sure at least one row is collected
                sample_row = group.iloc[[0]].copy()
                sample_row["conflict_vessel_count"] = vessel_count
                conflict_rows.append(sample_row)
                mmsi_set.update(group["MMSI"].unique())
            elif mode == "full":
                group["conflict_vessel_count"] = vessel_count
                conflict_rows.append(group)
        else:
            skipped_groups += 1

    print(f"\n[Part C Summary]")
    print(f"  Included groups: {included_groups}")
    print(f"  Skipped groups (vessel count outside allowed range): {skipped_groups}")
    print(f"  Conflict rows collected: {sum(len(grp) for grp in conflict_rows)}")

    # Final return
    if mode == "mmsi-only":
        return mmsi_set
    elif conflict_rows:
        return pd.concat(conflict_rows).reset_index(drop=True)
    else:
        return pd.DataFrame()
