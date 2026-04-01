import json
import torch
import logging

from pathlib import Path
from ultralytics import YOLO
from .utils import load_config

logger = logging.getLogger(__name__)

def get_flops_macs_absolute(model, imgsz=640):
    """
    Calculate absolute FLOPs and MACs using torch.profiler.

    Args:
        model (nn.Module): The model to profile.
        imgsz (int or list): Input image size (H, W).

    Returns:
        tuple: (FLOPs, MACs) as integers
    """
    p = next(model.parameters())  # YOLO stores nn.Module in .model
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]

    im = torch.empty((1, p.shape[1], *imgsz), device=p.device)

    with torch.profiler.profile(with_flops=True) as prof:
        model(im)

    flops = sum(x.flops for x in prof.key_averages())  # absolute FLOPs
    macs = flops // 2  # absolute MACs

    return flops, macs

def test_model():
    """Test the model using configurations from yaml file"""
    # Load testing configurations
    config = load_config('testing')

    logger.info("=== Starting Model Testing ===")
    try:
        # Load the trained model
        model = YOLO(config['model_path'])
        
        # Calculate FLOPs and MACs
        flops, macs = get_flops_macs_absolute(model, imgsz=config.get('imgsz', 640))
        logger.info(f"FLOPs: {flops}, MACs: {macs}")

        # Run validation on the test set
        val_config = {k: v for k, v in config.items() if k != 'model_path'}
        results = model.val(**val_config)

        # Prepare metrics dictionary
        metrics_dict = results.results_dict
        metrics_dict['flops'] = flops
        metrics_dict['macs'] = macs

        # Save metrics to JSON file
        output_path = Path(config['project']) / config['name'] / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

        return results

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()
