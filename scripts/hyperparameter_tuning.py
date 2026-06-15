import logging

from ultralytics import YOLO

from .utils import load_config

logger = logging.getLogger(__name__)


def tune_model():
    """Tune model hyperparameters using configurations from yaml file."""
    config = load_config("hyperparameter_tuning")

    logger.info("=== Starting Hyperparameter Tuning ===")
    logger.info("Model: %s", config["model"])
    logger.info("Dataset: %s", config["data"])
    logger.info("Epochs: %s", config["epochs"])
    logger.info("Iterations: %s", config["iterations"])

    try:
        model = YOLO(config["model"])

        search_space = config.get("search_space", {})
        if not search_space:
            raise ValueError("No search space defined in config.yaml")

        logger.info("Search space ranges:")
        for param, bounds in search_space.items():
            logger.info("  %s: %s", param, bounds)

        tune_config = {key: tuple(bounds) for key, bounds in search_space.items()}
        optimizer = config.get("optimizer", "AdamW")

        logger.info("Starting hyperparameter search with optimizer=%s", optimizer)
        model.tune(
            data=config['data'],
            epochs=config['epochs'],
            iterations=config['iterations'],
            space=tune_config,
            optimizer=optimizer,
            plots=True,
            save=True,
            val=True,
        )
        logger.info("Hyperparameter tuning completed successfully!")
    except Exception as exc:
        logger.error("Error during hyperparameter tuning: %s", exc)
        raise


if __name__ == "__main__":
    tune_model()
