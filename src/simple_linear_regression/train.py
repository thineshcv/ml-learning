import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from .model import LinearRegressionModel
from torch import nn

def train(X_train, y_train, model_path="best_model.pth"):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    n_features = X_train.shape[1]
    model = LinearRegressionModel(n_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    mlflow.set_experiment("housing-linear-regression")
    with mlflow.start_run():
        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 50 == 0:
                print(f"Epoch [{epoch+1}/300], Loss: {loss.item():.4f}")

        mlflow.log_param("epochs", 300)
        mlflow.log_metric("train_loss", loss.item())
        mlflow.pytorch.log_model(model, "model")

    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
