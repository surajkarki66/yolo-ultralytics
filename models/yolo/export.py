import logging

from ultralytics import YOLO
from pathlib import Path
from .utils import load_config

logger = logging.getLogger(__name__)

def export_model():
    """Export the model using configurations from yaml file"""
    # Load export configurations
    config = load_config('export')
    
    logger.info("=== Starting Model Export ===")
    try:
        # Load the model
        model = YOLO(config['model_path'])
        logger.info(f"Loaded model from {config['model_path']}")
        
        # Prepare export arguments
        export_args = {
            'format': config.get('format', 'onnx'),  # Default to ONNX if not specified
            'imgsz': config.get('imgsz', 640),
            'keras': config.get('keras', False),
            'optimize': config.get('optimize', False),
            'half': config.get('half', False),
            'int8': config.get('int8', False),
            'dynamic': config.get('dynamic', False),
            'simplify': config.get('simplify', True),
            'opset': config.get('opset'),
            'workspace': config.get('workspace'),
            'nms': config.get('nms', False),
            'batch': config.get('batch', 1),
            'device': config.get('device'),
            'data': config.get('data'),
            'fraction': config.get('fraction', 1.0)
        }
        
        # Log export configuration
        logger.info("Export configuration:")
        for key, value in export_args.items():
            if value is not None:  # Only log non-None values
                logger.info(f"  {key}: {value}")
        
        # Export the model
        logger.info(f"\nExporting model to {export_args['format']} format...")
        results = model.export(**export_args)
        
        # Log export results
        if isinstance(results, (str, Path)):
            logger.info(f"Model exported successfully to: {results}")
        else:
            logger.info("Model exported successfully!")
            logger.info(f"Export results: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during model export: {str(e)}")
        raise

if __name__ == "__main__":
    export_model()
