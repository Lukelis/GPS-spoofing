import numpy as np
import pandas as pd
from utils import haversine

# === PART A: Location jump & unrealistic speed ===
def detect_part_a(vessel_df):
    if len(vessel_df) < 2:
        return None

    vessel_df = vessel_df.sort_values("timestamp").reset_index(drop=True)
    vessel_df["lat_prev"] = vessel_df["Latitude"].shift()
    vessel_df["lon_prev"] = vessel_df["Longitude"].shift()
    vessel_df["time_prev"] = vessel_df["timestamp"].shift()

    vessel_df.dropna(subset=["lat_prev", "lon_prev", "time_prev"], inplace=True)
    vessel_df["dist_km"] = haversine(
        vessel_df["lat_prev"], vessel_df["lon_prev"],
        vessel_df["Latitude"], vessel_df["Longitude"]
    )
    vessel_df["time_diff_sec"] = (vessel_df["timestamp"] - vessel_df["time_prev"]).dt.total_seconds()

    # Avoid divide-by-zero
    vessel_df = vessel_df[vessel_df["time_diff_sec"] > 0]

    vessel_df["speed_kmph"] = vessel_df["dist_km"] / (vessel_df["time_diff_sec"] / 3600)

    spoof_flags = (vessel_df["speed_kmph"] > 100) | \
                  ((vessel_df["dist_km"] > 50) & (vessel_df["time_diff_sec"] < 300))

    anomalies = vessel_df[spoof_flags]
    return anomalies[["MMSI", "timestamp", "Latitude", "Longitude", "speed_kmph", "dist_km"]] if not anomalies.empty else None

# === PART B: Speed/course consistency ===
def detect_part_b(vessel_df):
    if len(vessel_df) < 3:
        return None

    vessel_df = vessel_df.sort_values("timestamp").reset_index(drop=True)
    vessel_df["SOG_prev"] = vessel_df["SOG"].shift()
    vessel_df["COG_prev"] = vessel_df["COG"].shift()
    vessel_df["time_prev"] = vessel_df["timestamp"].shift()

    vessel_df.dropna(subset=["SOG", "SOG_prev", "COG", "COG_prev", "time_prev"], inplace=True)
    vessel_df["time_diff_sec"] = (vessel_df["timestamp"] - vessel_df["time_prev"]).dt.total_seconds()

    # Avoid divide-by-zero
    vessel_df = vessel_df[vessel_df["time_diff_sec"] > 0]

    # Speed fluctuation: detect sudden large speed changes
    vessel_df["sog_change"] = (vessel_df["SOG"] - vessel_df["SOG_prev"]).abs()
    speed_jump_flag = (vessel_df["sog_change"] > 15) & (vessel_df["time_diff_sec"] < 120)

    # Course inconsistency: large turn angles in short time
    vessel_df["cog_change"] = (vessel_df["COG"] - vessel_df["COG_prev"]).abs()
    vessel_df["cog_change"] = vessel_df["cog_change"].apply(lambda x: 360 - x if x > 180 else x)
    course_jump_flag = (vessel_df["cog_change"] > 90) & (vessel_df["time_diff_sec"] < 120)

    anomalies = vessel_df[speed_jump_flag | course_jump_flag]
    return anomalies[["MMSI", "timestamp", "Latitude", "Longitude", "SOG", "COG", "sog_change", "cog_change"]] if not anomalies.empty else None

# === COMBINED: Run both A and B
def detect_spoofing(vessel_df):
    part_a = detect_part_a(vessel_df)
    part_b = detect_part_b(vessel_df)

    if part_a is not None and part_b is not None:
        combined = pd.concat([part_a, part_b]).drop_duplicates()
        return combined if not combined.empty else None
    elif part_a is not None:
        return part_a
    elif part_b is not None:
        return part_b
    else:
        return None
