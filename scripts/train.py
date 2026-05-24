import logging

from ultralytics import YOLO

from .utils import drop_keys, load_config

logger = logging.getLogger(__name__)


def train_model():
    """Train the model using configurations from yaml file."""
    config = load_config("training")

    logger.info("=== Starting Model Training ===")
    try:
        model = YOLO(config["model"])
        train_cfg = drop_keys(config, "model")
        results = model.train(**train_cfg)
        logger.info("Training completed successfully!")
        return results
    except Exception as exc:
        logger.error("Error during training: %s", exc)
        raise


if __name__ == "__main__":
    train_model()
