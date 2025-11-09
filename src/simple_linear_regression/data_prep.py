import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import os
import shutil

def download_data():
    """
    Downloads the housing dataset from Kaggle if it's not already present.
    """
    # Path where the dataset will be downloaded
    download_dir = "data"
    file_path = os.path.join(download_dir, "housing.csv")

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"Dataset already exists in {file_path}")
        return file_path

    # Download the dataset to the cache
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("huyngohoang/housingcsv")

    # Find the csv file in the downloaded dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".csv"):
                # Copy the file to the data directory
                shutil.copy(os.path.join(root, file), file_path)
                print(f"Dataset downloaded to {file_path}")
                return file_path

    raise Exception("Could not find the downloaded dataset.")

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Example: drop rows with missing values and select features
    df = df.dropna()
    y = df['Price'].values   # Replace with your target column
    X = df.drop(['Price', 'Address'], axis=1).select_dtypes(include=['float64', 'int64']).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
