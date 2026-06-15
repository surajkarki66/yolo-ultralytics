import torch
import logging

from pathlib import Path
from ultralytics import YOLO

from .utils import drop_keys, load_config, require_file, save_json

logger = logging.getLogger(__name__)


def get_flops_macs_absolute(yolo_model, imgsz=640):
    """
    Calculate absolute FLOPs and MACs for the underlying nn.Module.

    Args:
        yolo_model: Ultralytics YOLO wrapper.
        imgsz: Input image size (int or [H, W]).

    Returns:
        tuple: (FLOPs, MACs) as integers.
    """
    nn_model = yolo_model.model
    nn_model.eval()

    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]

    device = next(nn_model.parameters()).device
    channels = next(nn_model.parameters()).shape[1]
    im = torch.empty((1, channels, *imgsz), device=device)

    with torch.profiler.profile(with_flops=True) as prof:
        nn_model(im)

    flops = sum(item.flops for item in prof.key_averages())
    macs = flops // 2
    return flops, macs


def test_model():
    """Test the model using configurations from yaml file."""
    config = load_config("testing")
    require_file(config["model_path"], "Model checkpoint")

    logger.info("=== Starting Model Testing ===")
    try:
        model = YOLO(config["model_path"])

        flops, macs = get_flops_macs_absolute(model, imgsz=config.get("imgsz", 640))
        logger.info("FLOPs: %s, MACs: %s", flops, macs)

        val_config = drop_keys(config, "model_path")
        results = model.val(**val_config)

        metrics_dict = dict(results.results_dict)
        metrics_dict["flops"] = flops
        metrics_dict["macs"] = macs

        output_path = Path(config["project"]) / config["name"] / "metrics.json"
        save_json(metrics_dict, output_path)
        logger.info("Saved metrics to %s", output_path)

        return results
    except Exception as exc:
        logger.error("Error during testing: %s", exc)
        raise


if __name__ == "__main__":
    test_model()
