import pandas as pd
from src.logger import logger

class DataValidation:
    def __init__(self, config):
        self.config = config
        self.data_path = config["data_ingestion"]["raw_data_path"]
        self.target_column = config["data_validation"]["target_column"]
        self.report_path = config["data_validation"]["validation_report_path"]

    def validate_data(self):
        logger.info("Starting data validation")
        try:
            # Sometimes the CSV has extra spaces or encoding issues
            df = pd.read_csv(self.data_path, encoding='utf-8', engine='python')
            
            # Basic checks
            report = {}
            report["num_rows"] = df.shape[0]
            report["num_columns"] = df.shape[1]
            report["missing_values"] = df.isnull().sum().to_dict()
            report["target_unique_values"] = df[self.target_column].nunique()

            logger.info(f"Data validation report: {report}")

            # Save report
            from src.utils.common import save_json
            save_json(self.report_path, report)

            logger.info(f"Validation report saved at {self.report_path}")
            return df, report

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise e
