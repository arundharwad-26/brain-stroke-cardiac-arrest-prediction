# src/data_loader.py

import pandas as pd


def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    Returns a pandas DataFrame.
    """

    try:
        data = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully")
        print(f"📊 Shape of data: {data.shape}")
        print(f"🧾 Columns: {list(data.columns)}")
        return data

    except FileNotFoundError:
        print("❌ File not found. Please check the path.")
        return None

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


if __name__ == "__main__":
    # For testing purpose
    file_path = "C:\\Users\\arund\\Desktop\\Stroke_and_Cardiac\\health_data.csv"
    df = load_data(file_path)
