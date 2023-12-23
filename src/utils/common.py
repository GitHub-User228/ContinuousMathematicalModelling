"""
common.py

Purpose:
    Contains common functionalities used across the project.
"""

import os
import sys
import yaml
import json
from pathlib import Path

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src import logger



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file, and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the yaml file.

    Raises:
        ValueError: If the yaml file is empty.
        e: If any other exception occurs.

    Returns:
        ConfigBox: The yaml content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        logger.info("Value exception: empty yaml file")
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.info(f"An exception {e} has occurred")
        raise e



@ensure_annotations
def save_yaml(path: Path, data: dict):
    """
    Save yaml data

    Args:
        path (Path): path to yaml file
        data (dict): data to be saved in yaml file
    """
    try:
        with open(path, "w") as f:
            yaml.dump(data, f, indent=4)
        logger.info(f"yaml file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save yaml to {path}. Error: {e}")
        raise



@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")
    except PermissionError:
        logger.error(f"Permission denied to write to {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to save json to {path}. Error: {e}")
        raise



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    try:
        with open(path, "r") as f:
            content = json.load(f)
        logger.info(f"json file loaded successfully from: {path}")
        return ConfigBox(content)
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from {path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied to read {path}")
        raise
    except OSError as e:
        logger.error(f"Failed to load json from {path}. Error: {e}")
        raise