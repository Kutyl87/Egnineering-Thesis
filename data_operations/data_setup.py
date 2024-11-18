import os
import logging

from aws_handler.aws_s3_handler import AWSS3Handler
from data_operations.preprocessing.add_headers import DataParser


class DataSetup:
    def __init__(self, bucket_name: str):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
        self.aws_s3_handler = AWSS3Handler(bucket_name)
        self.data_parser = DataParser()

    def _load_data(self, processed_data_path: str, raw_data_path: str, filename: str) -> None:
        if os.path.exists(os.path.join(processed_data_path, filename)):
            self.logger.info("Data is available. There is no need to download it again")
        else:
            self.aws_s3_handler.download_file(filename, raw_data_path)

    def _preprocess_dataset(self, processed_data_path: str, raw_data_path: str) -> None:
        files = self.data_parser.get_csv_files(raw_data_path)
        self.data_parser.prepend_headers(files, processed_data_path)

    def prepare_data(self, processed_data_path: str, raw_data_path: str, filename: str) -> None:
        self._load_data(processed_data_path, raw_data_path, filename)
        self._preprocess_dataset(processed_data_path, raw_data_path)
