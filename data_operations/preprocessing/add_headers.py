import os
import csv
import logging


class DataParser:
    def __init__(self, encoding: str = "ascii"):
        self.encoding = encoding
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def get_csv_files(self, directory: str) -> list[str]:
        return [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.endswith(".csv")
        ]

    def _read_data(self, file: str) -> list[str]:
        with open(file, "r", newline="") as f:
            existing_data = f.readlines()
            return existing_data

    def _merge_data(
        self, file: str, previous_data: list[str], headers: list[str]
    ) -> None:
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            self.logger.info(f"Prepending headers to {file}")
            writer.writerow(headers)
            f.writelines(previous_data)

    def prepend_headers(self, file_list: list[str], headers: list[str]) -> None:
        for file in file_list:
            existing_data = self._read_data(file)
            if existing_data[0].strip().split(",") != headers:
                self._merge_data(file, existing_data, headers)
            else:
                self.logger.info(f"{file} has proper headers")


if __name__ == "__main__":
    data_parser = DataParser()
    files = data_parser.get_csv_files("../data")
    headers = [
        "Timestamp",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "I",
        "V",
        "PID set value",
        "DMC set value",
        "data set-value",
        "FLU",
        "FRU",
        "FLB",
        "FRB",
        "HL",
        "HR",
    ]
    data_parser.prepend_headers(files, headers)
