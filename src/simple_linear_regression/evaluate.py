import torch

def evaluate(X_test, y_test, model_path="best_model.pth"):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = torch.mean((predictions - y_test) ** 2).item()
        print(f"Test MSE: {mse:.4f}")
