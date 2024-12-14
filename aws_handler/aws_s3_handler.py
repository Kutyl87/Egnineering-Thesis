import boto3
import os
import logging


class AWSS3Handler:
    def __init__(self, aws_bucket_name: str) -> None:
        self.aws_bucket_name = aws_bucket_name
        self.s3_client = boto3.client("s3")
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def download_file(self, filename: str, destination_path: str) -> None:
        self._check_if_folder_exists(destination_path)
        self._check_if_file_exists(filename, destination_path)
        self.log_overriding_file(filename, destination_path)
        self.s3_client.download_file(
            self.aws_bucket_name, filename, f"{destination_path}/{filename}"
        )

    @staticmethod
    def _check_if_folder_exists(destination_path: str) -> None:
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

    def _check_if_file_exists(self, filename: str, destination_path: str) -> bool:
        is_file = os.path.exists(os.path.join(destination_path, filename))
        self.logger.info(f"File exists: {is_file}")
        return is_file

    def log_overriding_file(self, filename: str, destination_path: str) -> None:
        file_path = os.path.join(destination_path, filename)
        self.logger.info(f"Overriding the file {file_path}")


if __name__ == "__main__":
    AWSS3Handler("data-hvac").download_file(
        "OUT20_new_closed_loop_FIXED_2.csv", "./data/external"
    )
