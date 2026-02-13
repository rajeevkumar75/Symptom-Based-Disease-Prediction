import pandas as pd
from pathlib import Path
from src.utils.common import read_yaml, save_yaml
from src.logger import logger
import sys
import os
import pandas as pd
from dataclasses import dataclass

from src.logger import logger
from src.utils.common import read_yaml, create_directories
from src.utils.common import save_json
from src.logger import CustomException


class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_dataset(self, df: pd.DataFrame) -> bool:
        try:
            validation_status = True
            report = {}

            # 1️⃣ Dataset not empty
            if df.empty:
                validation_status = False
                report["dataset_empty"] = True
            else:
                report["dataset_empty"] = False

            # 2️⃣ Target column exists
            if self.config.target_column not in df.columns:
                validation_status = False
                report["target_column_exists"] = False
            else:
                report["target_column_exists"] = True

            # 3️⃣ Target column validation (multi-class)
            target_col = df[self.config.target_column]

            if target_col.isnull().sum() > 0:
                validation_status = False
                report["target_null_values"] = True
            else:
                report["target_null_values"] = False

            report["number_of_classes"] = target_col.nunique()

            # 4️⃣ Feature validation (binary check)
            feature_columns = df.drop(columns=[self.config.target_column])

            non_binary_columns = []
            for col in feature_columns.columns:
                unique_vals = set(feature_columns[col].unique())
                if not unique_vals.issubset({0, 1}):
                    non_binary_columns.append(col)

            if non_binary_columns:
                validation_status = False
                report["non_binary_columns"] = non_binary_columns
            else:
                report["non_binary_columns"] = "All binary ✔"

            # 5️⃣ Duplicate rows
            duplicate_count = df.duplicated().sum()
            report["duplicate_rows"] = int(duplicate_count)

            if duplicate_count > 0:
                logger.warning(f"Found {duplicate_count} duplicate rows")

            # Save report
            save_yaml(
                path=Path(self.config.validation_report_path),
                data=report
            )

            logger.info("Data validation completed successfully")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
