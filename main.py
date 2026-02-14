from src.config.configuration import Configuration
from src.components.model_trainer import ModelTrainer
from src.logger import logger

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Model Training Pipeline")

        # 1️⃣ Load configuration
        config_file_path = "config/config.yaml"
        config = Configuration(config_file_path)

        # 2️⃣ Initialize Model Trainer
        model_trainer = ModelTrainer(config)

        # 3️⃣ Train models
        best_model_path, metrics_path = model_trainer.train()

        logger.info("✅ Model training completed successfully")
        logger.info(f"🏆 Best model saved at: {best_model_path}")
        logger.info(f"📊 Metrics saved at: {metrics_path}")

    except Exception as e:
        logger.exception("❌ Model training pipeline failed")
        raise e
