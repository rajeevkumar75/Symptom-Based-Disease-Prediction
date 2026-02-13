from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    config_file_path = "config/config.yaml" 
    pipe = DataIngestion(config_path=config_file_path)
    pipe.download_data()