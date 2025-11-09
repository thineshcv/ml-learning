import sys
import os

# Add src to the Python path
sys.path.append(os.path.abspath('.'))

from src.simple_linear_regression.data_prep import download_data, load_and_preprocess
from src.simple_linear_regression.train import train
from src.simple_linear_regression.evaluate import evaluate


def main():
    # Download data
    csv_path = download_data()

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

    # Train model
    model_path = "best_model.pth"
    train(X_train, y_train, model_path)

    # Evaluate model
    evaluate(X_test, y_test, model_path)


if __name__ == "__main__":
    main()
