import pandas as pd
from pathlib import Path
from typing import Optional

from src.config.configuration import Configuration
from src.utils.common import load_object
from src.logger import logger


class PredictionPipeline:
    def __init__(self, config: Optional[Configuration] = None):
        if config is None:
            config = Configuration()
        self.config = config
        trainer_config = config.get_model_trainer_config()
        transformation_config = config.get_data_transformation_config()
        self.model_path = Path(trainer_config["model_dir"]) / trainer_config["best_model_name"]
        self.label_encoder_path = Path(transformation_config["label_encoder_path"])
        self.target_column = config.get_data_validation_config()["target_column"]
        logger.info(f"Loading model from {self.model_path}")
        self.model = load_object(self.model_path)
        logger.info(f"Loading label encoder from {self.label_encoder_path}")
        self.label_encoder = load_object(self.label_encoder_path)

    def predict(self, input_df: pd.DataFrame):
        predictions = self.model.predict(input_df)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        return predicted_labels
