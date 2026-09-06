# Calibration data

`vai_q_yolo.py` loads images from **`val_ids.txt`** (one absolute path per line). Supported extensions are listed in `IMG_FORMATS` inside `vai_q_yolo.py` (jpg, png, bmp, etc.).

## Layout

```text
data/
  val/
    img001.jpg
    img002.png
    ...
  val_ids.txt    # generated — do not edit by hand unless needed
```

## Generate the file list

From `vitis-ai/quantization/`:

```bash
python prepare_calibration_data.py
```

The script scans `data/val/`, writes absolute paths to `data/val_ids.txt`, and prints the image count.

## Tips

- Use enough images that match your deployment domain; calibration quality affects INT8 accuracy.
- Re-run `prepare_calibration_data.py` after adding, removing, or moving images.
- Image size in `val/` can vary; `vai_q_yolo.py` resizes to `--img_height` and `--img_width` during calibration.
