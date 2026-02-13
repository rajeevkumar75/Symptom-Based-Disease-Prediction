from dataclasses import dataclass
from pathlib import Path
from src.utils.common import read_yaml, create_directories


#fordataingestion
@dataclass
class DataIngestionConfig:
    raw_data_path: Path
    gdrive_folder_id: str
    file_name: str


#fordataValidation
@dataclass
class DataValidationConfig:
    target_column: str
    validation_report_path: Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = Path("config/config.yaml")
    ):
        self.config = read_yaml(config_filepath)

        create_directories([
            self.config.data_ingestion.artifacts_dir,
            self.config.data_ingestion.raw_data_dir,
            self.config.data_validation.validation_dir
        ])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        return DataIngestionConfig(
            raw_data_path=Path(config.raw_data_path),
            gdrive_folder_id=config.gdrive_folder_id,
            file_name=config.file_name
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        return DataValidationConfig(
            target_column=config.target_column,
            validation_report_path=Path(config.validation_report_path)
        )
