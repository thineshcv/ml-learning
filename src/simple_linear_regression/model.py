from torch import nn
import torch

class LinearRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        # Ensure input is a float32 torch.Tensor to avoid dtype mismatches
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.float32:
                x = x.to(dtype=torch.float32)
        else:
            # Accept lists / numpy arrays / pandas DataFrame from pyfunc->pandas path
            try:
                x = torch.as_tensor(x, dtype=torch.float32)
            except Exception:
                x = torch.tensor(x).float()
        return self.linear(x)