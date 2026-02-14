import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from src.utils.common import read_yaml, save_yaml, save_object
from src.config.configuration import Configuration
from src.logger import logger

class DataTransformation:
    def __init__(self, config: Configuration):
        self.config = config.get_data_transformation_config()
        self.raw_data_path = config.get_data_ingestion_config()["raw_data_path"]
        self.target_column = config.get_data_validation_config()["target_column"]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning...")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.fillna(0, inplace=True)
        logger.info(f"Data cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Encoding target column: {self.target_column}")
        le = LabelEncoder()
        df[self.target_column] = le.fit_transform(df[self.target_column])
        return df, le

    def initiate_data_transformation(self):
        try:
            logger.info("Starting Data Transformation Pipeline...")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Raw data loaded from: {self.raw_data_path}")

            #cleaned data
            df = self.clean_data(df)
            df, label_encoder = self.encode_target(df)

            Path(self.config["transformed_dir"]).mkdir(parents=True, exist_ok=True)

            #saving cleaned data
            cleaned_data_path = self.config["cleaned_data_path"]
            df.to_csv(cleaned_data_path, index=False)
            logger.info(f"Cleaned data saved at: {cleaned_data_path}")

            #saving label encoder
            le_path = self.config["label_encoder_path"]
            save_object(le_path, label_encoder)
            logger.info(f"Label encoder saved at: {le_path}")

            return cleaned_data_path, le_path

        except Exception as e:
            logger.exception("Error in Data Transformation")
            raise e
