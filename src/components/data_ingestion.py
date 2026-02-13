import os
import sys
import gdown

from src.logger import logger, CustomException
from src.utils.common import read_yaml, create_directories


class DataIngestion:
    def __init__(self, config_path: str):
        try:
            self.config = read_yaml(config_path)
            self.ingestion_config = self.config["data_ingestion"]

            self.raw_data_dir = self.ingestion_config["raw_data_dir"]
            self.gdrive_folder_id = self.ingestion_config["gdrive_folder_id"]

        except Exception as e:
            raise CustomException(e, sys)

    def download_data(self):
        try:
            logger.info("Starting data ingestion")

            create_directories([self.raw_data_dir])

            logger.info("Downloading dataset folder from Google Drive")

            gdown.download_folder(
                id=self.gdrive_folder_id,
                output=self.raw_data_dir,
                quiet=False,
                use_cookies=False
            )

            logger.info(f"Dataset folder downloaded into {self.raw_data_dir}")
            return self.raw_data_dir

        except Exception as e:
            raise CustomException(e, sys)
