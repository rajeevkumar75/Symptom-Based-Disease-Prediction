import unittest
from pathlib import Path

import pandas as pd

from src.config.configuration import Configuration
from src.components.model_trainer import ModelTrainer
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline


class TestTrainingAndPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Configuration("config/config.yaml")
        transformation_config = cls.config.get_data_transformation_config()
        cls.cleaned_data_path = transformation_config["cleaned_data_path"]
        cls.cleaned_df = pd.read_csv(cls.cleaned_data_path)
        validation_config = cls.config.get_data_validation_config()
        cls.target_column = validation_config["target_column"]

    def test_model_training_creates_artifacts(self):
        trainer = ModelTrainer(self.config)
        best_model_path, metrics_path = trainer.initiate_model_training()
        self.assertTrue(Path(best_model_path).is_file())
        self.assertTrue(Path(metrics_path).is_file())

    def test_prediction_pipeline_returns_labels(self):
        trainer = ModelTrainer(self.config)
        trainer_config = self.config.get_model_trainer_config()
        model_path = Path(trainer_config["model_dir"]) / trainer_config["best_model_name"]
        if not model_path.is_file():
            trainer.initiate_model_training()
        feature_df = self.cleaned_df.drop(columns=[self.target_column]).head(5)
        pipeline = PredictionPipeline(self.config)
        predictions = pipeline.predict(feature_df)
        self.assertEqual(len(predictions), len(feature_df))
        self.assertTrue(all(pred is not None for pred in predictions))

    def test_training_pipeline_instantiates(self):
        pipeline = TrainingPipeline("config/config.yaml")
        self.assertIsInstance(pipeline, TrainingPipeline)


if __name__ == "__main__":
    unittest.main()
