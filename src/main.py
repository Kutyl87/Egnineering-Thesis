import torch
import torch.nn as nn
import torch.optim as optim
from src.training_orchestrator import ModelOrchestrator

from data_operations.csv_data_loader import CSVDataLoader
from mlops.src.MLFlowManager import MLFlowManager

from ml_models.models.lstm_net import LSTMNet
from src.ml_pipeline import MLPipeline


def main():
    csv_loader = CSVDataLoader()
    input_size = 8
    hidden_size = 28
    num_layers = 1
    out_size = 5
    seq_length = 50
    model = LSTMNet(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, out_size=out_size, seq_length=seq_length)
    orchestrator = ModelOrchestrator(model, csv_loader)
    mlflow_manager = MLFlowManager(experiment_name="LSTM experiment", tracking_uri="http://127.0.0.1:5000")
    pipeline = MLPipeline(orchestrator, mlflow_manager)
    data_paths = ("./data/processed", "./data/raw")
    filename = "OUT20_new_closed_loop_FIXED_2.csv"
    batch_size = 32
    epochs = 2
    learning_rate = 0.01
    model_name = "LSTM basic model "
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pipeline.run_training_pipeline(data_paths, filename, batch_size, loss_function, optimizer, epochs, model_name)
    # eval_loss = pipeline.run_evaluation_pipeline(data_paths, filename, batch_size, loss_function)
    # predictions = pipeline.run_prediction_pipeline(data_paths, filename, batch_size)
    # mlflow_manager.log_metrics()


if __name__ == "__main__":
    main()
