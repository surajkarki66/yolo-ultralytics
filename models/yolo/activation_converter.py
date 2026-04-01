import torch
import sys
import os
import logging

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_activation_function(model):
    """
    Recursively convert SiLU activation functions to Hardswish.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of activations replaced
    """
    count = 0
    
    def _iter(module, prefix=''):
        nonlocal count
        for name, sub_module in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(sub_module, torch.nn.SiLU) and sub_module.__class__.__name__ == "SiLU":
                setattr(module, name, torch.nn.Hardswish())
                logger.info(f"Replaced SiLU with Hardswish in layer: {full}")
                count += 1
            _iter(sub_module, prefix=full)
    
    _iter(model)
    return count


def convert_model(input_path, output_dir, output_name):
    """
    Convert a YOLO model's activation functions to Hardswish and save it.
    
    Args:
        input_path: Path to input .pt file
        output_dir: Directory to save converted model
        output_name: Name for output file (without .pt extension)
        
    Returns:
        Path to saved model
    """
    logger.info(f"Loading model from: {input_path}")
    logger.info(f"Target activation: hardswish")
    
    # Load FULL checkpoint
    checkpoint = torch.load(input_path, weights_only=False)
    
    # Extract model from checkpoint
    model = checkpoint["model"]
    model = model.float().to("cpu")
    
    # Convert SiLU to Hardswish
    logger.info("=" * 60)
    logger.info("Converting SiLU activations to Hardswish...")
    logger.info("=" * 60)
    count = convert_activation_function(model)
    logger.info("=" * 60)
    logger.info(f"Conversion complete! Replaced {count} SiLU layers with Hardswish")
    logger.info("=" * 60)
    
    # Put the converted model back into the checkpoint
    checkpoint["model"] = model
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the model
    output_path = os.path.join(output_dir, f"{output_name}.pt")
    torch.save(checkpoint, output_path)
    
    logger.info(f"Model successfully saved to: {output_path}")
    
    return output_path


def main():
    """Command-line interface for the converter."""
    if len(sys.argv) != 4:
        print(__doc__)
        print("\nError: Incorrect number of arguments")
        print("\nUsage:")
        print("  python -m models.yolo.activation_converter <input_model.pt> <output_dir> <output_name>")
        print("\nArguments:")
        print("  input_model.pt  : Path to the input YOLO model (.pt file)")
        print("  output_dir      : Directory where converted model will be saved")
        print("  output_name     : Name for the output file (without .pt extension)")
        print("\nExample:")
        print("  python -m models.yolo.activation_converter runs/train/exp/weights/best.pt models/assets my_model")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    output_name = sys.argv[3]
    
    try:
        convert_model(input_path, output_dir, output_name)
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()