from src.constants import *
from src.utils.common import read_yaml
from src import logger
from src.entity.config_entity import OptimizatorConfig



class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for reading and providing 
    configuration settings needed for various stages of the data pipeline.

    Attributes:
    - config (dict): Dictionary holding configuration settings from the config file.
    - params (dict): Dictionary holding parameter values from the params file.
    """
    
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        """
        Initializes the ConfigurationManager with configurations, parameters, and schema.

        Parameters:
        - config_filepath (str): Filepath to the configuration file.
        - params_filepath (str): Filepath to the parameters file.
        """
        self.config = self._read_config_file(config_filepath, "config")
        self.params = self._read_config_file(params_filepath, "params")

    
    def _read_config_file(self, filepath: str, config_name: str) -> dict:
        """
        Reads and returns the content of a configuration file.

        Parameters:
        - filepath (str): The file path to the configuration file.
        - config_name (str): Name of the configuration (used for logging purposes).

        Returns:
        - dict: Dictionary containing the configuration settings.

        Raises:
        - Exception: An error occurred reading the configuration file.
        """
        try:
            return read_yaml(filepath)
        except Exception as e:
            logger.error(f"Error reading {config_name} file: {filepath}. Error: {e}")
            raise

    
    def get_optimizator_config(self) -> OptimizatorConfig:
        """
        Extracts and returns optimizator configuration settings as a OptimizatorConfig object.

        Returns:
        - OptimizatorConfig: Object containing optimizator configuration settings.

        Raises:
        - AttributeError: The 'optimizator' attribute does not exist in the config file.
        """
        try:
            config = self.config.optimizator
            return OptimizatorConfig(
                data = config.data,
                n_runs = config.n_runs,
                model_name = config.model_name,
                metric_name = config.metric_name,
                optimizator_name = config.optimizator_name,
                test_ratio = config.test_ratio,
                dt = config.dt,
                seed = config.seed,
                path_to_results = config.path_to_results
            )
        except AttributeError as e:
            logger.error("The 'optimizator' attribute does not exist in the config file.")
            raise e