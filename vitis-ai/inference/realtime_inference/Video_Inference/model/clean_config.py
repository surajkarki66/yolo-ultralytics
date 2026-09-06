#!/usr/bin/env python3
"""
Clean YOLO config pickle file by removing DFL objects and converting tensors to numpy arrays.

Usage:
    python clean_config.py --input config.pkl --output config_cleaned.pkl
    python clean_config.py -i config.pkl -o config_cleaned.pkl
"""

import argparse
import pickle
import torch
import numpy as np
from pathlib import Path


# -------------------------------
# Helper functions
# -------------------------------
def is_dfl(obj):
    """Check if object is DFL class."""
    return hasattr(obj, '__class__') and obj.__class__.__name__ == "DFL"


def convert_tensor_to_numpy(obj):
    """Convert PyTorch tensors to numpy arrays."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj


# -------------------------------
# Recursively remove DFL objects and convert tensors
# -------------------------------
def clean_config(obj):
    """Recursively remove DFL objects and convert tensors to numpy arrays."""
    # Remove DFL objects
    if is_dfl(obj):
        return None
    
    # Convert tensor to numpy first
    obj = convert_tensor_to_numpy(obj)
    
    # Tuple
    if isinstance(obj, tuple):
        cleaned = []
        for x in obj:
            # Skip DFL objects
            if is_dfl(x):
                continue
            cleaned_x = clean_config(x)
            if cleaned_x is not None:
                cleaned.append(cleaned_x)
        return tuple(cleaned) if cleaned else None
    
    # List
    if isinstance(obj, list):
        cleaned = []
        for x in obj:
            if is_dfl(x):
                continue
            cleaned_x = clean_config(x)
            if cleaned_x is not None:
                cleaned.append(cleaned_x)
        return cleaned if cleaned else None
    
    # Dict
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if is_dfl(v):
                continue
            cleaned_v = clean_config(v)
            if cleaned_v is not None:
                cleaned[k] = cleaned_v
        return cleaned if cleaned else None
    
    # Other objects remain unchanged
    return obj


def clean_config_file(input_path: str, output_path: str, verbose: bool = True):
    """
    Clean a config pickle file by removing DFL objects and converting tensors.
    
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
        data = pickle.load(f)
    
    if verbose:
        print("Loaded successfully")
        print(f"Original data type: {type(data)}")
        if isinstance(data, tuple) and len(data) > 0:
            print(f"Original structure (first 2 elements):")
            for i, item in enumerate(data[:2]):
                if isinstance(item, torch.Tensor):
                    print(f"  Element {i}: torch.Tensor shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"  Element {i}: {type(item).__name__} = {item}")
    
    # Clean config (remove DFL and convert tensors)
    if verbose:
        print("\nRemoving DFL objects and converting tensors to numpy arrays...")
    
    cleaned = clean_config(data)
    
    if verbose:
        print("Cleaning completed successfully")
        print(f"Cleaned data type: {type(cleaned)}")
        if isinstance(cleaned, tuple) and len(cleaned) > 0:
            print(f"Cleaned structure (first 2 elements):")
            for i, item in enumerate(cleaned[:2]):
                if isinstance(item, np.ndarray):
                    print(f"  Element {i}: numpy.ndarray shape={item.shape}, dtype={item.dtype}, values={item}")
                else:
                    print(f"  Element {i}: {type(item).__name__} = {item}")
    
    # Save
    if verbose:
        print(f"\nSaving to {output_path}...")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "wb") as f:
        pickle.dump(cleaned, f)
    
    if verbose:
        print("Saved successfully")
    
    return cleaned


def verify_file(file_path: str, verbose: bool = True):
    """Verify that the cleaned config file can be loaded properly."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if verbose:
        print("\n=== Verification ===")
    
    with open(file_path, "rb") as f:
        loaded = pickle.load(f)
    
    if verbose:
        print(f"Successfully loaded: {type(loaded)}")
        if isinstance(loaded, tuple):
            print(f"Tuple length: {len(loaded)}")
            for i, item in enumerate(loaded):
                if isinstance(item, np.ndarray):
                    print(f"  Element {i}: numpy.ndarray shape={item.shape}, dtype={item.dtype}, values={item}")
                elif isinstance(item, torch.Tensor):
                    print(f"  Element {i}: torch.Tensor (WARNING - should be converted!) shape={item.shape}")
                else:
                    item_type = type(item).__name__
                    if hasattr(item, '__class__'):
                        class_name = item.__class__.__name__
                        print(f"  Element {i}: {item_type} (class: {class_name}) = {item}")
                    else:
                        print(f"  Element {i}: {item_type} = {item}")
        
        # Check for any remaining tensors
        def check_for_tensors(obj, path=""):
            if isinstance(obj, torch.Tensor):
                print(f"  ⚠️ Found tensor at {path}: shape={obj.shape}")
                return True
            elif isinstance(obj, (tuple, list)):
                found = False
                for i, item in enumerate(obj):
                    if check_for_tensors(item, f"{path}[{i}]"):
                        found = True
                return found
            elif isinstance(obj, dict):
                found = False
                for k, v in obj.items():
                    if check_for_tensors(v, f"{path}[{k}]"):
                        found = True
                return found
            return False
        
        if check_for_tensors(loaded):
            print("\n⚠️ WARNING: Some tensors were not converted to numpy arrays!")
        else:
            print("\n✅ No PyTorch tensors found - all converted to numpy arrays")
    
    return loaded


def main():
    parser = argparse.ArgumentParser(
        description="Clean YOLO config pickle file by removing DFL objects and converting tensors to numpy arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s -i config.pkl -o config_cleaned.pkl
    %(prog)s --input yolov8n_B4096_config.pkl --output yolov8n_B4096_config_cleaned.pkl
    %(prog)s -i config.pkl -o config_cleaned.pkl --quiet
    %(prog)s -i config.pkl -o config_cleaned.pkl --verify
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
        
        # Print summary
        if verbose:
            print("\n" + "="*50)
            print(f"✅ Successfully cleaned config file!")
            print(f"   Input:  {args.input}")
            print(f"   Output: {args.output}")
            print("="*50)
        else:
            print(f"✅ Successfully cleaned {args.input} -> {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())