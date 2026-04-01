import logging

from ultralytics.utils.benchmarks import benchmark
from .utils import load_config

logger = logging.getLogger(__name__)

def benchmark_model():
    """Benchmark the model using configurations from yaml file"""
    # Load benchmark configurations
    config = load_config('benchmark')
    
    logger.info("=== Starting Model Benchmarking ===")
    try:
        # Run benchmark with all configured parameters
        results = benchmark(
            data=config['data'],
            model=config['model_path'],
            imgsz=config['imgsz'],
            half=config['half'],
            device=config['device'],
            verbose=config['verbose'],
            int8=config['int8'],
            format=config['format']
        )
        
        logger.info("Benchmarking completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Error during benchmarking: {str(e)}")
        raise

if __name__ == "__main__":
    benchmark_model()

    

