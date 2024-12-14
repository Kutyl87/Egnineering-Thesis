import os
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

from data.torch_datasets.sequential_dataset import SequentialDataset
from data_operations.csv_data_loader import CSVDataLoader
from data_operations.data_setup import DataSetup
import torch.nn as nn

from ml_models.training.training_utils import train_model

load_dotenv()


class ModelOrchestrator:
    def __init__(self, model: nn.Module, data_loader: CSVDataLoader):
        self.model = model
        self.data_loader = data_loader

    def prepare_data(self,
                     processed_data_path: str,
                     raw_data_path: str,
                     filename: str) -> None:
        self.data_loader.prepare_data(processed_data_path, raw_data_path, filename)

    def get_data_loaders(self, processed_data_path: str, filename: str, batch_size: int) -> tuple[
        DataLoader, DataLoader]:
        train_dataset, test_dataset = self.data_loader.get_data_in_dataset_train(processed_data_path, filename)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_eval_loader(self, processed_data_path: str, filename: str, batch_size: int) -> DataLoader:
        eval_dataset = self.get_eval_dataset(processed_data_path, filename)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return eval_loader

    def get_eval_dataset(self, processed_data_path: str, filename: str) -> SequentialDataset:
        eval_dataset = self.data_loader.get_data_in_dataset_eval(processed_data_path, filename)
        return eval_dataset

    def eval_model(self, data_loader: DataLoader,
                   loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], device: torch.device) -> float:
        num_batches = len(data_loader)
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                output = self.model(X)
                total_loss += loss_function(output, y).item()
        avg_loss = total_loss / num_batches
        return avg_loss

    def train_model(self, data_loader: DataLoader,
                    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    optimizer: torch.optim.Optimizer, device: torch.device) -> float:
        num_batches = len(data_loader)
        total_loss = 0
        self.model.train()
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = self.model(X)
            loss = loss_function(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        return avg_loss

    def predict(self, data_loader: DataLoader, device: torch.device) -> torch.Tensor:
        output = torch.tensor([], device=device)
        self.model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                X = X.to(device)
                y_star = self.model(X)
                output = torch.cat((output, y_star), 0)
        return output
