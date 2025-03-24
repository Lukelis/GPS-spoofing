import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from spoofing_detector import detect_spoofing

def run_parallel_detection(df, processes=None):
    """
    Run GPS spoofing detection in parallel for all vessels in the dataset.

    Args:
        df (pd.DataFrame): Cleaned AIS dataset.
        processes (int, optional): Number of worker processes. Defaults to all available CPUs.

    Returns:
        pd.DataFrame: Combined spoofing anomalies from all vessels.
    """
    if processes is None:
        processes = cpu_count()

    grouped = [group for _, group in df.groupby("MMSI")]

    print(f"Running detection on {len(grouped)} vessels using {processes} processes...")

    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(detect_spoofing, grouped), total=len(grouped)))

    # Filter out None and concatenate results
    anomalies_df = pd.concat([r for r in results if r is not None and not r.empty], ignore_index=True)
    return anomalies_df