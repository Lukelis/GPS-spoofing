from data_loader import load_ais_data
from spoofing_detector import detect_spoofing

# Step 1: Define input file
DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv" # Update path if needed

# Step 2: Load and preview AIS data
def main():
    df = load_ais_data(DATA_PATH)
    print("\nData sample:")
    print(df.head())

    # Step 3: Select one vessel to test spoofing detection
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