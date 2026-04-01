#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path

from models.yolo import train_model, test_model
from models.yolo.benchmark import benchmark_model
from models.yolo.export import export_model
from models.yolo.hyperparameter_tuning import tune_model
from dataset_utils.eda import perform_eda
from dataset_utils.visualize_dataset import visualize_sample_annotated_dataset
from models.yolo.cross_validation import run_cross_validation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    logger.info("=== Starting Training ===")
    try:
        # Train using the imported train module
        train_model()
        logger.info("Training completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during training: {str(e)}")
        return False

def run_testing():
    """Run testing using the imported test module"""
    logger.info("=== Starting Testing ===")
    try:   
        # Test using the imported test module
        test_model()
        logger.info("Testing completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during testing: {str(e)}")
        return False

def run_cross_validation_cmd():
    """Run cross-validation using the imported cross_validation module"""
    logger.info("=== Starting Cross-Validation ===")
    try:
        # Run cross-validation using the imported module
        results = run_cross_validation()
        print(results)
        logger.info("Cross-validation completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during cross-validation: {str(e)}")
        return False

def run_benchmark():
    """Run benchmarking using the imported benchmark module"""
    logger.info("=== Starting Benchmarking ===")
    try:
        # Benchmark using the imported benchmark module
        benchmark_model()
        logger.info("Benchmarking completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during benchmarking: {str(e)}")
        return False

def run_export():
    """Run model export using the imported export module"""
    logger.info("=== Starting Model Export ===")
    try:
        # Export using the imported export module
        export_model()
        logger.info("Model export completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during export: {str(e)}")
        return False

def run_tuning():
    """Run hyperparameter tuning using the imported tune module"""
    logger.info("=== Starting Hyperparameter Tuning ===")
    try:
        # Tune using the imported tune module
        tune_model()
        logger.info("Hyperparameter tuning completed successfully!")
        return True
    except Exception as e:
        print(e)
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        return False

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
                
            args.func(data_dir, output_dir)
            success = True
            
        sys.exit(0 if success else 1)
    except Exception as e:
        print(e)
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()

