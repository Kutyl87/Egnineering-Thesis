import os

import torch
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
from data_operations.data_setup import DataSetup
import torch.nn as nn

load_dotenv()


class ModelOrchestrator:
    def __init__(self, model: nn.Module, trainingUtils):
        self.model = model
        self.data_setup = DataSetup(os.getenv('AWS_BUCKET_NAME'))
        self.training_utils = trainingUtils

    def prepare_data(self,
                     processed_data_path: str,
                     raw_data_path: str,
                     filename: str) -> None:
        self.data_setup.prepare_data(processed_data_path, raw_data_path, filename)

