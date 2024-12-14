import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
import torch


def test_model(data_loader: DataLoader, model: nn.Module,
               loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(data_loader: DataLoader, model: nn.Module,
                loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                optimizer: torch.optim.Optimizer) -> float:
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss


def predict(data_loader: DataLoader, model: nn.Module) -> torch.Tensor:
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


def train_loop(data_loader: DataLoader, model: nn.Module,
               loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               optimizer: torch.optim.Optimizer, num_epochs: int) -> None:
    for epoch in range(num_epochs):
        avg_loss = train_model(data_loader, model, loss_function, optimizer)
        print(f'Epoch : {epoch} | Avg Loss : {avg_loss:}')
