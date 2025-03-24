from data_loader import load_ais_data
from spoofing_detector import detect_spoofing
from parallel_runner import run_parallel_detection

# Step 1: Define input file
DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv" # Update path if needed
RUN_ALL_VESSELS = True #Set to false if one vessel is needed for testing

# Step 2: Load and preview AIS data
def main():
    df = load_ais_data(DATA_PATH)
    print("\nData sample:")
    print(df.head())

    if RUN_ALL_VESSELS:
        print("\nRunning spoofing detection in parallel for all vessels...")
        anomalies = run_parallel_detection(df)

        if anomalies is not None and not anomalies.empty:
            print("\nTotal anomalies found:", len(anomalies))
            print(anomalies.head())
            anomalies.to_csv("spoofing_anomalies_output.csv", index=False)
            print("\nSaved to 'spoofing_anomalies_output.csv'")
        else:
            print("\nNo anomalies detected across vessels.")
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
