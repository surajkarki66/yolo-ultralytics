import logging

from ultralytics.utils.benchmarks import benchmark

from .utils import load_config, require_file

logger = logging.getLogger(__name__)


def benchmark_model():
    """Benchmark the model using configurations from yaml file."""
    config = load_config("benchmark")
    require_file(config["model_path"], "Model checkpoint")
    if config.get("data"):
        require_file(config["data"], "Dataset yaml")

    logger.info("=== Starting Model Benchmarking ===")
    try:
        results = benchmark(
            data=config["data"],
            model=config["model_path"],
            imgsz=config["imgsz"],
            half=config["half"],
            device=config["device"],
            verbose=config["verbose"],
            int8=config["int8"],
            format=config["format"],
        )
        logger.info("Benchmarking completed successfully!")
        return results
    except Exception as exc:
        logger.error("Error during benchmarking: %s", exc)
        raise


if __name__ == "__main__":
    benchmark_model()
