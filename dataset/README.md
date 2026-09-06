# Dataset Guide

This folder holds the YOLO-format dataset consumed by `main.py`.

## Expected Layout

```text
dataset/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/            
    images/
    labels/
```

## Label Format

Each label file is YOLO normalized format:

```text
class_id x_center y_center width height
```

- one object per line
- values normalized to `[0, 1]`
- class IDs must match `data.yaml` names

## YOLO OBB Label Format

For oriented bounding boxes (OBB), each line stores 4 corner points:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

- one object per line
- `(x1, y1) ... (x4, y4)` are the four box corners in order around the object
- coordinates are typically normalized to `[0, 1]` relative to image width/height
- class IDs must match `data.yaml` names

## Recommended Checks

```bash
python3 main.py eda --data-dir dataset --output-dir output
python3 main.py visualize --data-dir dataset --output-dir output --num-samples 10
```

These commands validate split consistency and generate quick visual sanity checks.