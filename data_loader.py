import pandas as pd

def load_ais_data(filepath):
    """
    Load and clean AIS data from a given CSV file.

    Args:
        filepath (str): Path to the AIS CSV file.

    Returns:
        pd.DataFrame: Cleaned AIS dataset.
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["# Timestamp"])
    df.rename(columns={"# Timestamp": "timestamp"}, inplace=True)

    # Filter out non-vessel records
    df = df[df["Type of mobile"] != "Base Station"]

    # Drop rows with missing coordinates
    df.dropna(subset=["Latitude", "Longitude"], inplace=True)

    # Sort by vessel and time
    df = df.sort_values(by=["MMSI", "timestamp"])
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} AIS records across {df['MMSI'].nunique()} vessels.")
    return df

