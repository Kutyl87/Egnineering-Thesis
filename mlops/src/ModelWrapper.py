import mlflow
import torch


class ModelWrapper(mlflow.pyfunc.PythonFunction):
    def __init__(self, model):
        self.model = model

    def predict(self, model_input):
        if isinstance(self.model, torch.nn.Module):
            model_input = torch.tensor(model_input.values).float()
            with torch.no_grad():
                return self.model(model_input).numpy()
        else:
            return self.model.predict(model_input)
