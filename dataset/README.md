# Dataset

This directory is intended for the YOLO-formatted dataset used for training and evaluation by the main pipeline (`main.py`) and by the Vitis AI evaluation scripts.

## Human Presence Detection Dataset

The Human Presence Detection Dataset is a combination of VOC2012_Person and a private collection of images annotated with bounding boxes for detecting people. This dataset is designed for training and evaluating object detection models, mainly YOLO.

### Dataset details

- **Content:** Images of people in various settings (e.g., indoor, outdoor, different poses, lighting conditions).
- **Sources:**
  - Public: PASCAL VOC 2012 Person dataset.
  - Private: Custom collection of images from Steinel GmbH.
- **Annotations:** Bounding boxes in YOLO format, with class ID `0` representing "person".
- **Image size:** Images are typically 640×640 pixels (adjust based on your dataset).
- **Annotation format:** Text files (`.txt`) with one line per bounding box: `class_id x_center y_center width height` (normalized to [0, 1]).

### Expected layout (YOLO format)

Place your dataset under `dataset/` (or point `--data-dir` to it) with a structure such as:

- `train/images/`, `train/labels/`
- `valid/images/`, `valid/labels/`
- `test/images/`, `test/labels/` (optional)

Use `main.py eda` and `main.py visualize` to inspect the data.
