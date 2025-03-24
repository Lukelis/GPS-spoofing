from data_loader import load_ais_data

# Step 1: Define input file
DATA_PATH = "C:/Users/lukan/Downloads/aisdk-2025-03-14/aisdk-2025-03-14.csv"

# Step 2: Load and preview AIS data
def main():
    df = load_ais_data(DATA_PATH)
    print("\nData sample:")
    print(df.head())

if __name__ == "__main__":
    main()
