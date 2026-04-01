import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(section: str, config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file for a specific section"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get(section, {})
    except FileNotFoundError:
        logging.error(f"Config file {config_path} not found!")
        raise
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise