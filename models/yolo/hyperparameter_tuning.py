import logging
from ultralytics import YOLO
from .utils import load_config

logger = logging.getLogger(__name__)

def tune_model():
    """Tune model hyperparameters using configurations from yaml file"""
    # Load tuning configurations
    config = load_config('hyperparameter_tuning')
    
    logger.info("=== Starting Hyperparameter Tuning ===")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Dataset: {config['data']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Iterations: {config['iterations']}")
    
    try:
        # Load model
        model = YOLO(config['model'])
        
        # Get search space from config
        search_space = config.get('search_space', {})
        if not search_space:
            raise ValueError("No search space defined in config.yaml")
        
        # Log search space ranges
        logger.info("\nSearch Space Ranges:")
        for param, (min_val, max_val) in search_space.items():
            logger.info(f"{param}: [{min_val}, {max_val}]")
        
        # Convert search space to tune format (tuple of min, max)
        tune_config = {k: tuple(v) for k, v in search_space.items()}
        print(tune_config)
        
        # Run hyperparameter tuning
        logger.info("\nStarting hyperparameter search...")
        model.tune(
            data=config['data'],
            epochs=config['epochs'],
            iterations=config['iterations'],
            space=tune_config,
            optimizer="AdamW",
            plots=True,
            save=True,
            val=True
        )
        logger.info("Hyperparameter tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        raise

if __name__ == "__main__":
    tune_model()
