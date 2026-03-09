from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import Configuration
from src.utils.common import read_yaml
from src.logger import logger


class TrainingPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = Configuration(config_path)

    def run(self):
        logger.info("Starting full training pipeline")

        logger.info("Running data ingestion")
        data_ingestion = DataIngestion(self.config_path)
        raw_data_dir = data_ingestion.download_data()
        logger.info(f"Data downloaded to {raw_data_dir}")

        logger.info("Running data validation")
        raw_config = read_yaml(self.config_path)
        data_validation = DataValidation(raw_config)
        _, _ = data_validation.validate_data()

        logger.info("Running data transformation")
        data_transformation = DataTransformation(self.config)
        cleaned_data_path, label_encoder_path = data_transformation.initiate_data_transformation()

        logger.info("Running model training")
        model_trainer = ModelTrainer(self.config)
        best_model_path, metrics_path = model_trainer.initiate_model_training()

        logger.info("Training pipeline completed successfully")

        validation_report_path = self.config.get_data_validation_config()["validation_report_path"]

        return {
            "raw_data_dir": str(Path(raw_data_dir)),
            "cleaned_data_path": str(cleaned_data_path),
            "label_encoder_path": str(label_encoder_path),
            "best_model_path": str(best_model_path),
            "metrics_path": str(metrics_path),
            "validation_report_path": str(validation_report_path),
        }
