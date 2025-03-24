from data_loader import load_ais_data
from spoofing_detector import detect_spoofing
from parallel_runner import run_parallel_detection
from neighbor_detector import detect_neighbor_conflicts  # <-- Make sure this is imported

# Step 1: Define input file
DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv" # Update path if needed
RUN_ALL_VESSELS = True
ANOMALY_THRESHOLD = 2

def main():
    df = load_ais_data(DATA_PATH)
    print("\nData sample:")
    print(df.head())

    if RUN_ALL_VESSELS:
        print("\nRunning spoofing detection in parallel for all vessels...")
        anomalies = run_parallel_detection(df)

        # === Filter by anomaly threshold
        vessel_counts = anomalies.groupby("MMSI").size()
        reliable_mmsis = vessel_counts[vessel_counts >= ANOMALY_THRESHOLD].index
        filtered_anomalies = anomalies[anomalies["MMSI"].isin(reliable_mmsis)]

        # === Summary statistics
        print(f"\nTotal raw anomalies found: {len(anomalies)}")
        print(f"Filtered anomalies (≥{ANOMALY_THRESHOLD} per vessel): {len(filtered_anomalies)}")

        total_rows = len(df)
        spoofed_rows = len(filtered_anomalies)
        percentage = (spoofed_rows / total_rows) * 100

        affected_vessels = filtered_anomalies["MMSI"].nunique()
        total_vessels = df["MMSI"].nunique()
        vessel_percentage = (affected_vessels / total_vessels) * 100

        print(f"Percentage of spoofed records: {percentage:.2f}%")
        print(f"Vessels with spoofing (≥{ANOMALY_THRESHOLD} anomalies): {affected_vessels}/{total_vessels} ({vessel_percentage:.2f}%)")

        if not filtered_anomalies.empty:
            print(filtered_anomalies.head())
            filtered_anomalies.to_csv("spoofing_anomalies_output.csv", index=False)
            print("\nSaved to 'spoofing_anomalies_output.csv'")
        else:
            print("\nNo vessels passed the anomaly threshold.")

        # === PART C: Neighbor Conflict Detection
        print("\nRunning Part C: Neighbor conflict detection (grid-based)...")
        try:
            # Only keep needed columns to save memory
            reduced_df = df[["timestamp", "Latitude", "Longitude", "MMSI"]].copy()
            neighbor_conflicts = detect_neighbor_conflicts(reduced_df, time_window='5min',grid_size=0.005)

            if not neighbor_conflicts.empty:
                print(f"Conflicting positions found: {len(neighbor_conflicts)} records")
                print(neighbor_conflicts.head())
                neighbor_conflicts.to_csv("neighbor_conflicts_output.csv", index=False)
                print("Saved to 'neighbor_conflicts_output.csv'")
            else:
                print("No conflicting positions detected in nearby vessels.")
        except Exception as e:
            print("Part C failed due to memory or data issue:")
            print(str(e))

    else:
        sample_mmsi = df["MMSI"].unique()[0]
        vessel_df = df[df["MMSI"] == sample_mmsi]

        print(f"\nRunning spoofing detection for MMSI: {sample_mmsi}")
        anomalies = detect_spoofing(vessel_df)

        if anomalies is not None and not anomalies.empty:
            print("\nAnomalies found:")
            print(anomalies.head())
        else:
            print("\nNo anomalies detected for this vessel.")

if __name__ == "__main__":
    main()
