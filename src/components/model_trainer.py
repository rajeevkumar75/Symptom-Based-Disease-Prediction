# src/components/model_trainer.py

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from src.logger import logger
from src.utils.common import save_object, save_json
from src.config.configuration import Configuration


class ModelTrainer:
    def __init__(self, config: Configuration):
        self.config = config
        self.trainer_config = config.get_model_trainer_config()
        self.target_column = config.get_data_validation_config()["target_column"]

    def _load_data(self):
        logger.info("Loading cleaned data for model training")
        df = pd.read_csv(self.trainer_config["cleaned_data_path"])

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        return train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def _get_models(self):
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(n_estimators=200),
            "NaiveBayes": BernoulliNB(),
            "SVM": SVC(kernel="rbf", probability=True),
        }

    def initiate_model_training(self):
        """
        Public method called from main.py
        """
        try:
            logger.info("🚀 Starting Model Training")

            X_train, X_test, y_train, y_test = self._load_data()
            models = self._get_models()

            metrics = {}
            best_model = None
            best_score = 0.0
            best_model_name = None

            for model_name, model in models.items():
                logger.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                metrics[model_name] = {
                    "accuracy": acc,
                    "classification_report": classification_report(
                        y_test, y_pred, output_dict=True
                    ),
                }

                logger.info(f"{model_name} accuracy: {acc:.4f}")

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = model_name

            # Create directories
            Path(self.trainer_config["model_dir"]).mkdir(parents=True, exist_ok=True)

            # Save best model
            best_model_path = (
                Path(self.trainer_config["model_dir"])
                / self.trainer_config["best_model_name"]
            )
            save_object(best_model_path, best_model)

            # Save metrics
            metrics_path = Path(self.trainer_config["metrics_path"])
            save_json(metrics_path, metrics)

            logger.info(f"🏆 Best model: {best_model_name}")
            logger.info(f"📦 Model saved at: {best_model_path}")
            logger.info(f"📊 Metrics saved at: {metrics_path}")

            return str(best_model_path), str(metrics_path)

        except Exception as e:
            logger.exception("❌ Error during model training")
            raise e
