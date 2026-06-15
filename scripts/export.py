import logging


from pathlib import Path
from ultralytics import YOLO

from .utils import load_config, require_file

logger = logging.getLogger(__name__)


def export_model():
    """Export the model using configurations from yaml file."""
    config = load_config("export")
    require_file(config["model_path"], "Model checkpoint")

    logger.info("=== Starting Model Export ===")
    try:
        model = YOLO(config["model_path"])
        logger.info("Loaded model from %s", config["model_path"])

        export_args = {
            "format": config.get("format", "onnx"),
            "imgsz": config.get("imgsz", 640),
            "keras": config.get("keras", False),
            "optimize": config.get("optimize", False),
            "half": config.get("half", False),
            "int8": config.get("int8", False),
            "dynamic": config.get("dynamic", False),
            "simplify": config.get("simplify", True),
            "opset": config.get("opset"),
            "workspace": config.get("workspace"),
            "nms": config.get("nms", False),
            "batch": config.get("batch", 1),
            "device": config.get("device"),
            "data": config.get("data"),
            "fraction": config.get("fraction", 1.0),
        }

        logger.info("Export configuration:")
        for key, value in export_args.items():
            if value is not None:
                logger.info("  %s: %s", key, value)

        logger.info("Exporting model to %s format...", export_args["format"])
        results = model.export(**export_args)

        if isinstance(results, (str, Path)):
            logger.info("Model exported successfully to: %s", results)
        else:
            logger.info("Model exported successfully!")
            logger.info("Export results: %s", results)

        return results
    except Exception as exc:
        logger.error("Error during model export: %s", exc)
        raise


if __name__ == "__main__":
    export_model()
