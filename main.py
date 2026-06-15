#!/usr/bin/env python3

import argparse
import sys
import logging

from pathlib import Path

from dataset_utils.eda import perform_eda
from dataset_utils.visualize_dataset import visualize_sample_annotated_dataset
from scripts import (
    benchmark_model,
    export_model,
    run_cross_validation,
    test_model,
    train_model,
    tune_model,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _run_action(action_name: str, func) -> bool:
    """Run a command action with consistent logging and error handling."""
    logger.info(f"=== Starting {action_name} ===")
    try:
        result = func()
        if result is not None:
            logger.info(result)
        logger.info(f"{action_name} completed successfully!")
        return True
    except Exception:
        logger.exception(f"Error during {action_name.lower()}")
        return False

def setup_parser():
    """Setup command line argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description='Dataset Analysis and Model Training Tools'
    )
    
    # Add subcommand parser
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Parser for eda
    parser_eda = subparsers.add_parser(
        'eda',
        help='Perform exploratory data analysis'
    )
    parser_eda.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the dataset directory'
    )
    parser_eda.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )
    parser_eda.set_defaults(func=perform_eda)

    # Parser for visualize
    parser_visualize = subparsers.add_parser(
        'visualize',
        help='Visualize sample annotated images'
    )
    parser_visualize.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the dataset directory'
    )
    parser_visualize.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )
    parser_visualize.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of sample annotated images to visualize (default: 5)'
    )
    parser_visualize.set_defaults(func=visualize_sample_annotated_dataset)

    # Parser for train
    parser_train = subparsers.add_parser(
        'train',
        help='Train model'
    )
    parser_train.set_defaults(func=run_training)

    # Parser for test
    parser_test = subparsers.add_parser(
        'test',
        help='Test model'
    )
    parser_test.set_defaults(func=run_testing)

    # Parser for cross-validation
    parser_cv = subparsers.add_parser(
        'cross-validation',
        help='Run k-fold cross-validation'
    )
    parser_cv.set_defaults(func=run_cross_validation_cmd)

    # Parser for benchmark
    parser_benchmark = subparsers.add_parser(
        'benchmark',
        help='Benchmark model performance'
    )
    parser_benchmark.set_defaults(func=run_benchmark)

    # Parser for export
    parser_export = subparsers.add_parser(
        'export',
        help='Export model to different formats'
    )
    parser_export.set_defaults(func=run_export)

    # Parser for tune
    parser_tune = subparsers.add_parser(
        'tune',
        help='Perform hyperparameter tuning'
    )
    parser_tune.set_defaults(func=run_tuning)

    return parser

def run_training():
    """Run training using the imported train module"""
    return _run_action("Training", train_model)

def run_testing():
    """Run testing using the imported test module"""
    return _run_action("Testing", test_model)

def run_cross_validation_cmd():
    """Run cross-validation using the imported cross_validation module"""
    return _run_action("Cross-Validation", run_cross_validation)

def run_benchmark():
    """Run benchmarking using the imported benchmark module"""
    return _run_action("Benchmarking", benchmark_model)

def run_export():
    """Run model export using the imported export module"""
    return _run_action("Model Export", export_model)

def run_tuning():
    """Run hyperparameter tuning using the imported tune module"""
    return _run_action("Hyperparameter Tuning", tune_model)

def main():
    """Main function"""
    
    # Setup and parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.action:
        parser.print_help()
        sys.exit(1)

    
    # Run the selected command
    try:
        if args.action in ['train', 'test', 'cross-validation', 'benchmark', 'export', 'tune']:
            success = args.func()
        else:
            # For dataset analysis commands
            data_dir = Path(args.data_dir)
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            if not data_dir.exists():
                logger.error(f"Dataset directory {data_dir} does not exist.")
                sys.exit(1)

            if args.action == 'visualize':
                args.func(data_dir, output_dir, num_samples=args.num_samples)
            else:
                args.func(data_dir, output_dir)
            success = True
            
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

