from src.config.configuration import Configuration
from src.components.model_trainer import ModelTrainer
from src.logger import logger

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Model Training Pipeline")
        config_file_path = "config/config.yaml"
        config = Configuration(config_file_path)
        model_trainer = ModelTrainer(config)
        best_model_path, metrics_path = model_trainer.initiate_model_training()
        logger.info("✅ Model training completed successfully")
        logger.info(f"🏆 Best model saved at: {best_model_path}")
        logger.info(f"📊 Metrics saved at: {metrics_path}")
    except Exception as e:
        logger.exception("❌ Model training pipeline failed")
        raise e
