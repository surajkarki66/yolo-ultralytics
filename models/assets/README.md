# Model assets

This directory stores model-related assets used by the training, testing, benchmark, and export pipeline, such as:

- Checkpoint weights (e.g. `best.pt`, `last.pt`) produced by `python3 main.py train`
- Exported model files (e.g. ONNX, TensorRT engine) produced by `python3 main.py export`

Keep lightweight metadata/examples here, and exclude large weights via `.gitignore` or store them in an artifact/model registry.
