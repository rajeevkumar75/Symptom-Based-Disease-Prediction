from pathlib import Path
from src.utils.common import read_yaml, save_yaml
from src.logger import logger

class Configuration:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml"
    ):
        self.config_path = config_path
        self.params_path = params_path

        self.config = read_yaml(self.config_path)
        self.params = read_yaml(self.params_path)

        self.artifacts_root = self.config.get("artifacts_root", "artifacts")

    # --------------------- Data Ingestion Config ---------------------
    def get_data_ingestion_config(self):
        di_config = self.config.get("data_ingestion", {})
        return {
            "source_type": di_config.get("source_type"),
            "gdrive_folder_id": di_config.get("gdrive_folder_id"),
            "file_name": di_config.get("file_name"),
            "raw_data_dir": Path(di_config.get("raw_data_dir", "artifacts/raw_data")),
            "raw_data_path": Path(di_config.get("raw_data_path", "artifacts/raw_data/symptoms-diseases.csv"))
        }

    # --------------------- Data Validation Config ---------------------
    def get_data_validation_config(self):
        dv_config = self.config.get("data_validation", {})
        return {
            "target_column": dv_config.get("target_column", "prognosis"),
            "validation_dir": Path(dv_config.get("validation_dir", "artifacts/data_validation")),
            "validation_report_path": Path(dv_config.get("validation_report_path", "artifacts/data_validation/validation_report.json"))
        }


    # --------------------- Data Transformation Config ---------------------
    def get_data_transformation_config(self):
        dt_config = self.config.get("data_transformation", {})
        return {
            "transformed_dir": Path(dt_config.get("transformed_dir", "artifacts/data_transformation")),
            "cleaned_data_path": Path(dt_config.get("cleaned_data_path", "artifacts/data_transformation/cleaned_data.csv")),
            "label_encoder_path": Path(dt_config.get("label_encoder_path", "artifacts/data_transformation/label_encoder.pkl"))
        }
    

    # ---------------- Model Trainer ----------------
    def get_model_trainer_config(self):
        mt = self.config.get("model_trainer", {})

        return {
            "trained_model_dir": Path(mt.get("trained_model_dir", "artifacts/model_trainer")),
            "model_report_path": Path(mt.get("model_report_path", "artifacts/model_trainer/model_report.yaml")),
            "params": self.params
        }
