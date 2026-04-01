# Calibration data for Vitis AI quantization

This directory holds the calibration dataset used by `vai_q_yolo.py` for post-training quantization.

## Expected structure

```
data/
├── val/           # Validation/calibration images
└── val_ids.txt    # List of image paths or IDs used for calibration
```

Place your calibration images under `val/` and ensure `val_ids.txt` lists the samples to use. You can generate or copy these using `prepare_calibration_data.py` from the parent `quantization/` directory.
