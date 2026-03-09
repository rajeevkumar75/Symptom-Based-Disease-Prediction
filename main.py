from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import logger


def run_training_pipeline(config_path: str = "config/config.yaml") -> None:
    """
    Run the full end-to-end pipeline:
    - data ingestion
    - data validation
    - data transformation -> produces artifacts/data_transformation/cleaned_data.csv
    - model training using the cleaned data

    All input/output paths are taken from the configuration file.
    """
    pipeline = TrainingPipeline(config_path=config_path)
    summary = pipeline.run()

    logger.info("✅ End-to-end training pipeline finished successfully")
    logger.info(f"Raw data directory: {summary['raw_data_dir']}")
    logger.info(f"Cleaned data path (used for training): {summary['cleaned_data_path']}")
    logger.info(f"Label encoder path: {summary['label_encoder_path']}")
    logger.info(f"Best model path: {summary['best_model_path']}")
    logger.info(f"Metrics path: {summary['metrics_path']}")
    logger.info(f"Validation report path: {summary['validation_report_path']}")


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting end-to-end training pipeline")
        run_training_pipeline()
    except Exception:
        logger.exception("❌ End-to-end training pipeline failed")
        raise

