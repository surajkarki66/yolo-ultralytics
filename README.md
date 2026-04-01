# Efficient Execution of Object Detection Algorithms on Edge Devices

This project provides a comprehensive toolkit for training, evaluating, and deploying YOLO (You Only Look Once) object detection models, with a focus on optimizing performance for edge devices. It includes scripts for exploratory data analysis, data visualization, model training, testing, cross-validation, hyperparameter tuning, export to various formats, and deployment via **AMD Vitis AI** (quantization and compilation for FPGA).

## 1. Project Structure

```
.
├── config.yaml
├── custom_layers/
│   ├── block.py
│   └── conv.py
├── dataset/
│   └── README.md
├── dataset_utils/
│   ├── __init__.py
│   ├── eda.py
│   └── visualize_dataset.py
├── main.py
├── models/
│   ├── assets/
│   │   └── README.md
│   └── yolo/
│       ├── __init__.py
│       ├── activation_converter.py
│       ├── benchmark.py
│       ├── cross_validation.py
│       ├── export.py
│       ├── hyperparameter_tuning.py
│       ├── test.py
│       ├── train.py
│       └── utils.py
├── patch_ultralytics.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── vitis-ai/
    ├── compilation/
    │   ├── Architectures/
    │   ├── README.md
    │   ├── run_compile.sh
    │   ├── YOLOv8/
    │   └── YOLOv8-OBB/
    ├── evaluation/
    │   ├── Detect/
    │   │   ├── coco_prep.py
    │   │   ├── evaluate.py
    │   │   ├── models/
    │   │   ├── post_processing.py
    │   │   ├── test_data/
    │   │   ├── utils.py
    │   │   └── visualize_output.py
    │   ├── OBB/
    │   │   ├── evaluate.py
    │   │   ├── post_processing.py
    │   │   ├── remove_dfl.py
    │   │   └── test_data/
    │   ├── README.md
    │   └── pyproject.toml
    └── quantization/
        ├── custom_layers/
        ├── data/
        │   └── README.md
        ├── README.md
        ├── patch_ultralytics.py
        ├── prepare_calibration_data.py
        ├── run_compression.sh
        ├── run_yolo_conversion.sh
        ├── vai_q_yolo.py
        └── yolo_converter.py
```

## 2. Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Install dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## 3. Dataset

The project uses a YOLO-formatted dataset, which should be placed in the `dataset` directory. The dataset structure is expected to be in YOLO format with images and labels subdirectories for train, valid, and test splits. See `dataset/README.md` for details.

## 4. Usage

The main entry point for all functionalities is `main.py`. It uses a command-line interface with several subcommands.

### 4.1. Exploratory Data Analysis (EDA)

To perform EDA on the dataset, run:

```bash
python3 main.py eda --data-dir <path-to-dataset> --output-dir <path-to-output>
```

### 4.2. Visualize Dataset

To visualize a few sample images with their annotations, run:

```bash
python3 main.py visualize --data-dir <path-to-dataset> --output-dir <path-to-output> --num-samples 10
```

### 4.3. Training

To train the model, run:

```bash
python3 main.py train
```

The training configuration can be modified in `config.yaml` under the `training` section.

### 4.4. Testing

To test the trained model, run:

```bash
python3 main.py test
```

The testing configuration can be modified in `config.yaml` under the `testing` section.

### 4.5. Cross-Validation

To perform k-fold cross-validation, run:

```bash
python3 main.py cross-validation
```

The cross-validation configuration can be modified in `config.yaml` under the `cross_validation` section.

### 4.6. Hyperparameter Tuning

To perform hyperparameter tuning, run:

```bash
python3 main.py tune
```

The hyperparameter tuning configuration can be modified in `config.yaml` under the `hyperparameter_tuning` section.

### 4.7. Benchmark

To benchmark the model's performance, run:

```bash
python3 main.py benchmark
```

The benchmark configuration can be modified in `config.yaml` under the `benchmark` section.

### 4.8. Export

To export the model to a different format (e.g., ONNX, TFLite), run:

```bash
python3 main.py export
```

The export configuration can be modified in `config.yaml` under the `export` section.

## 5. Vitis AI Workflow

For edge deployment on AMD FPGAs:

1. **Quantization** — See `vitis-ai/quantization/README.md` for SiLU→HardSwish conversion, Ultralytics patching, and running Vitis AI quantization.
2. **Compilation** — See `vitis-ai/compilation/README.md` for compiling quantized models for the target DPU.
3. **Evaluation** — See `vitis-ai/evaluation/README.md` for evaluating quantized/compiled models (Detect and OBB).

## 6. Configuration

All configurations for the different steps are centralized in the `config.yaml` file. This file is divided into sections for `training`, `testing`, `cross_validation`, `hyperparameter_tuning`, `benchmark`, and `export`.
