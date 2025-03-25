from data_loader import load_ais_data
from spoofing_detector import detect_spoofing
from parallel_runner import run_parallel_detection
from neighbor_detector import detect_neighbor_conflicts
import pandas as pd

# Step 1: Define input file
DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv"
RUN_ALL_VESSELS = True
ANOMALY_THRESHOLD = 10

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

        print(f"\nTotal raw anomalies found: {len(anomalies)}")
        print(f"Filtered anomalies (≥{ANOMALY_THRESHOLD} per vessel): {len(filtered_anomalies)}")

        total_rows = len(df)
        spoofed_rows = len(filtered_anomalies)
        percentage = (spoofed_rows / total_rows) * 100

        affected_vessels_ab = filtered_anomalies["MMSI"].nunique()
        total_vessels = df["MMSI"].nunique()
        vessel_percentage_ab = (affected_vessels_ab / total_vessels) * 100

        print(f"Percentage of spoofed records (A+B): {percentage:.2f}%")
        print(f"Vessels with spoofing (≥{ANOMALY_THRESHOLD} anomalies): {affected_vessels_ab}/{total_vessels} ({vessel_percentage_ab:.2f}%)")

        if not filtered_anomalies.empty:
            print(filtered_anomalies.head())
            filtered_anomalies.to_csv("spoofing_anomalies_output.csv", index=False)
            print("\nSaved to 'spoofing_anomalies_output.csv'")
        else:
            print("\nNo vessels passed the anomaly threshold.")

        # === PART C: Neighbor Conflict Detection
        print("\nRunning Part C: Neighbor conflict detection (grid-based)...")
        try:
            reduced_df = df[["timestamp", "Latitude", "Longitude", "MMSI"]].copy()
            neighbor_conflicts = detect_neighbor_conflicts(
                reduced_df,
                time_window='2min',
                grid_size=0.002,
                max_vessels=15,
                mode="light"
            )

            if isinstance(neighbor_conflicts, pd.DataFrame) and not neighbor_conflicts.empty:
                print(f"Conflicting positions found: {len(neighbor_conflicts)} records")

                # Filter Part C to MMSIs with ≥2 conflict events
                c_counts = neighbor_conflicts.groupby("MMSI").size()
                reliable_c_mmsis = c_counts[c_counts >= 3].index
                neighbor_conflicts = neighbor_conflicts[neighbor_conflicts["MMSI"].isin(reliable_c_mmsis)]

                print(f"Filtered neighbor conflict vessels (≥3 records): {neighbor_conflicts['MMSI'].nunique()}")
                neighbor_conflicts.to_csv("neighbor_conflicts_output.csv", index=False)
                print("Saved to 'neighbor_conflicts_output.csv'")
            else:
                print("Conflict detection ran successfully, but no groups passed the filter.")

            # === Merge A+B with C
            print("\nMerging spoofing anomalies with neighbor conflicts...")

            mmsis_ab = set(filtered_anomalies["MMSI"])
            mmsis_c = set(neighbor_conflicts["MMSI"])

            combined_df = pd.concat([filtered_anomalies, neighbor_conflicts], ignore_index=True)
            combined_df["source"] = combined_df.apply(
                lambda row: "A_or_B" if "speed_kmph" in row or "sog_change" in row else "C",
                axis=1
            )
            combined_df = combined_df.drop_duplicates(subset=["MMSI", "timestamp"])

            # === Combined stats
            total_spoofed_vessels = combined_df["MMSI"].nunique()
            overlap = len(mmsis_ab & mmsis_c)
            only_ab = len(mmsis_ab - mmsis_c)
            only_c = len(mmsis_c - mmsis_ab)

            print(f"\n[Merge Stats]")
            print(f"  Vessels only in A+B: {only_ab}")
            print(f"  Vessels only in C  : {only_c}")
            print(f"  Vessels in both    : {overlap}")

            combined_df.to_csv("final_spoofing_combined.csv", index=False)
            print("\nSaved combined anomalies (A+B+C) to 'final_spoofing_combined.csv'")
            print(f"→ Total unique spoofing records: {len(combined_df)}")
            print(f"→ Total vessels involved: {total_spoofed_vessels}/{total_vessels} ({100 * total_spoofed_vessels / total_vessels:.2f}%)")

        except Exception as e:
            print("Part C failed due to memory or data issue:")
            print(str(e))

    else:
        # === Single vessel testing
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
