# main.py

from src.config.configuration import Configuration
from src.components.data_transformation import DataTransformation
from src.logger import logger

if __name__ == "__main__":
    try:
        logger.info("Starting Data Transformation Pipeline...")

        # 1️⃣ Load configuration
        config_file_path = "config/config.yaml"
        config = Configuration(config_file_path)

        # 2️⃣ Initialize DataTransformation
        data_transformer = DataTransformation(config)

        # 3️⃣ Run transformation
        cleaned_data_path, label_encoder_path = data_transformer.initiate_data_transformation()

        logger.info("Data Transformation completed successfully")
        logger.info(f"Cleaned data saved at: {cleaned_data_path}")
        logger.info(f"Label encoder saved at: {label_encoder_path}")

    except Exception as e:
        logger.exception("Data Transformation pipeline failed")
        raise e
