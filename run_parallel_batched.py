import multiprocessing as mp
import pandas as pd
from spoofing_detector import detect_spoofing

def process_vessel_batch(vessel_dfs):
    results = []
    for vessel_df in vessel_dfs:
        result = detect_spoofing(vessel_df)
        if result is not None and not result.empty:
            results.append(result)
    return pd.concat(results) if results else pd.DataFrame()

def run_parallel_batched(df, batch_size=100, num_workers=None):
    """
    Optimized: pre-splits the data by vessel, then batches vessel groups.

    Args:
        df (pd.DataFrame): Full AIS dataset
        batch_size (int): Number of vessels per worker batch
        num_workers (int): Number of parallel processes

    Returns:
        pd.DataFrame with all detected anomalies
    """
    grouped = [v for _, v in df.groupby("MMSI")]
    batches = [grouped[i:i + batch_size] for i in range(0, len(grouped), batch_size)]

    with mp.Pool(processes=num_workers or mp.cpu_count()) as pool:
        results = pool.map(process_vessel_batch, batches)

    return pd.concat([r for r in results if not r.empty]).reset_index(drop=True)
