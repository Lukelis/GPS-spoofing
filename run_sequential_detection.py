from spoofing_detector import detect_spoofing
import pandas as pd

def run_sequential_detection(df):
    """
    Run spoofing detection for each vessel sequentially (no multiprocessing).

    Args:
        df (pd.DataFrame): Full AIS dataset

    Returns:
        pd.DataFrame: All detected anomalies from all vessels
    """
    results = []
    grouped = df.groupby("MMSI")

    for mmsi, vessel_df in grouped:
        result = detect_spoofing(vessel_df)
        if result is not None and not result.empty:
            results.append(result)

    if results:
        return pd.concat(results).reset_index(drop=True)
    else:
        return None
