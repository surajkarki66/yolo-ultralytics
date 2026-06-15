#!/usr/bin/env python3
"""
Clean YOLO config pickle file by removing DFL/Identity objects and converting tensors to numpy arrays.

Usage:
    python clean_config.py --input config.pkl --output config_cleaned.pkl
    python clean_config.py -i config.pkl -o config_cleaned.pkl
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path


def is_dfl(obj):
    """Check if object is DFL class."""
    return hasattr(obj, '__class__') and obj.__class__.__name__ == "DFL"


def is_identity(obj):
    """Check if object is Identity class."""
    return hasattr(obj, '__class__') and obj.__class__.__name__ == "Identity"


def clean_config(config):
    """Clean config: remove DFL/Identity, convert tensors to numpy."""
    if not isinstance(config, tuple):
        return config
    
    cleaned = []
    for item in config:
        # Skip DFL and Identity
        if is_dfl(item) or is_identity(item):
            continue
        
        # Convert tensor to numpy
        if isinstance(item, torch.Tensor):
            cleaned.append(item.detach().cpu().numpy())
        # Handle nested tuples
        elif isinstance(item, tuple):
            nested = clean_config(item)
            if nested is not None:
                cleaned.append(nested)
        else:
            cleaned.append(item)
    
    return tuple(cleaned) if cleaned else None


def clean_config_file(input_path: str, output_path: str, verbose: bool = True):
    """
    Clean a config pickle file.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to output pickle file
        verbose: Print detailed information
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Check if input exists
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load original
    if verbose:
        print(f"Loading {input_path}...")
    with open(input_file, "rb") as f:
        original = pickle.load(f)
    
    if verbose:
        print(f"Original config: {original}")
        print(f"\nOriginal types:")
        for i, item in enumerate(original):
            print(f"  Element {i}: {type(item)} - {item}")
    
    # Clean config
    if verbose:
        print("\nCleaning config (removing DFL, Identity, converting tensors)...")
    cleaned = clean_config(original)
    
    if verbose:
        print(f"\nCleaned config: {cleaned}")
        print(f"\nCleaned types:")
        for i, item in enumerate(cleaned):
            print(f"  Element {i}: {type(item)} - {item}")
    
    # Save
    if verbose:
        print(f"\nSaving to {output_path}...")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump(cleaned, f)
    
    if verbose:
        print("✓ Done!")
    
    return cleaned


def verify_file(file_path: str, verbose: bool = True):
    """Verify that a cleaned config file can be loaded properly."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if verbose:
        print("\n=== Verification ===")
    
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    
    if verbose:
        print(f"Successfully loaded: {loaded}")
        print(f"Config length: {len(loaded)}")
        
        for i, item in enumerate(loaded):
            if isinstance(item, np.ndarray):
                print(f"Element {i}: numpy array shape={item.shape}, dtype={item.dtype}, values={item}")
            else:
                print(f"Element {i}: type={type(item)}, value={item}")
    
    return loaded


def main():
    parser = argparse.ArgumentParser(
        description="Clean YOLO config pickle file by removing DFL/Identity objects and converting tensors to numpy arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s -i config.pkl -o config_cleaned.pkl
    %(prog)s --input yolov26_config.pkl --output yolov26_config_cleaned.pkl
    %(prog)s -i config.pkl -o config_cleaned.pkl --quiet
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input pickle file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output pickle file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the output file after cleaning"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        # Clean the config file
        cleaned = clean_config_file(args.input, args.output, verbose)
        
        # Verify if requested
        if args.verify:
            verify_file(args.output, verbose)
        
        print(f"\n✅ Successfully cleaned {args.input} -> {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())