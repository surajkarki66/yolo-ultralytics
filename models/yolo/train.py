import logging
from ultralytics import YOLO
from .utils import load_config

logger = logging.getLogger(__name__)

def train_model():
    """Train the model using configurations from yaml file"""
    # Load training configurations
    config = load_config('training')
    
    logger.info("=== Starting Model Training ===")
    try:
        # Load model
        model = YOLO(config['model'])
        
        # Train the model with configuration parameters
        results = model.train(**config)
        logger.info("Training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()

