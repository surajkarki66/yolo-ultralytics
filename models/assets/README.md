# Model assets

This directory is a placeholder for model-related assets used by the training and evaluation pipeline, such as:

- Pre-trained or checkpoint weights (e.g. `best.pt`, `last.pt`) produced by `main.py train`
- Exported model files (e.g. ONNX, TFLite) from `main.py export`

Keep this directory in version control; large weight files can be excluded via `.gitignore` and stored separately or in a model registry.
