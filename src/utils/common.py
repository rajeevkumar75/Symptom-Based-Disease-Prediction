import os
import sys
import yaml
import json
import pickle
from pathlib import Path

from src.logger import logger, CustomException



def read_yaml(file_path: str):
    """
    Reads YAML file and returns content
    """
    try:
        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {file_path}")
            return content
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_list: list):
    """
    Create directories if not exist
    """
    try:
        for path in path_list:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj):
    """
    Save Python object as pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logger.info(f"Object saved at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load pickle object
    """
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logger.info(f"Object loaded from: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)
    
def save_json(path: str, data: dict) -> None:
    """
    Save dictionary data as a JSON file.

    Args:
        path (str): File path where JSON will be saved
        data (dict): Data to be saved
    """
    try:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"JSON file saved successfully at: {path}")

    except Exception as e:
        raise CustomException(e)
    

def save_yaml(file_path: str, content: dict) -> None:
    """
    Save a Python dictionary as a YAML file.

    Args:
        file_path (str): Path where YAML file will be saved
        content (dict): Dictionary content to save
    """
    try:
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)

        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file, default_flow_style=False)

        logger.info(f"YAML file saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

