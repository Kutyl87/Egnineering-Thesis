import logging
import os
from typing import Tuple

import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from aws_handler.aws_s3_handler import AWSS3Handler
from data.torch_datasets.sequential_dataset import SequentialDataset
from data_operations.data_setup import DataSetup
import torch.utils.data as data
import pandas as pd

from data_operations.feature_engineering.feature_transformation import FeatureTransformation

load_dotenv()


class CSVDataLoader:
    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.data_setup = DataSetup(os.getenv('AWS_BUCKET_NAME'))
        self.f

    def prepare_data(self,
                     processed_data_path: str,
                     raw_data_path: str,
                     filename: str) -> None:
        self.data_setup.prepare_data(processed_data_path, raw_data_path, filename)

    def get_data_in_df(self, processed_data_path: str, filename: str) -> pd.DataFrame:
        if os.path.exists(os.path.join(processed_data_path, filename)):
            self.logger.error(f"Loading a file {os.path.join(processed_data_path, filename)}")
            return pd.read_csv(os.path.join(processed_data_path, filename))
        raise FileNotFoundError("File does not exist.")

    def get_train_test_data(self, processed_data_path: str, filename) -> pd.DataFrame:
        return self.get_data_in_df(processed_data_path, filename)

    def _feature_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_transformer = FeatureTransformation(df)
        return feature_transformer.drop_columns(["PID set value", "DMC set value", "data set-value"])

    def _split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X, y = self._get_x_y_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def _get_x_y_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        X = df[["I", "V", "FLU", "FRU", "FLB", "FRB", "HL", "HR"]]
        y = df[["T1", "T2", "T3", "T4", "T5"]]
        return X, y

    def _build_dataset(self, x: pd.DataFrame, y: pd.DataFrame) -> SequentialDataset:
        return SequentialDataset(x.values, y.values)

    def get_data_in_dataset_train(self, processed_data_path: str, filename: str) -> tuple[
        SequentialDataset, SequentialDataset]:
        df_data = self.get_data_in_df(processed_data_path, filename)
        transformed_df = self._feature_transformation(df_data)
        X_train, X_test, y_train, y_test = self._split_data(transformed_df)
        train_dataset = self._build_dataset(X_train, y_train)
        test_dataset = self._build_dataset(X_test, y_test)
        return train_dataset, test_dataset

    def get_data_in_dataset_eval(self, processed_data_path: str, filename: str) -> SequentialDataset:
        df_data = self.get_data_in_df(processed_data_path, filename)
        transformed_df = self._feature_transformation(df_data)
        X, y = self._get_x_y_data(transformed_df)
        return SequentialDataset(X, y)
