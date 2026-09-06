from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys

import cv2

_DETECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _DETECT_DIR not in sys.path:
    sys.path.insert(0, _DETECT_DIR)


def _class_bgr(class_id: int) -> tuple[int, int, int]:
    """Distinct BGR color per class index (golden-ratio hue spacing)."""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.92)
    return int(b * 255), int(g * 255), int(r * 255)


def _load_class_labels(gt_path: str | None) -> dict[int, str] | None:
    """Map YOLO class index -> short label from gt_eval.json (digit names or single class)."""
    if not gt_path or not os.path.isfile(gt_path):
        return None
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories") or []
    out: dict[int, str] = {}
    for c in cats:
        name = str(c.get("name", ""))
        if name.isdigit():
            out[int(name)] = name
    if not out and len(cats) == 1:
        out[0] = str(cats[0].get("name", "?"))[:24]
    return out if out else None


def _label_for_class(yolo_cls: int, gt_labels: dict[int, str] | None) -> str:
    if gt_labels is not None and yolo_cls in gt_labels:
        return gt_labels[yolo_cls]
    return str(yolo_cls)


def _parse_pred_file(path: str) -> list[tuple[str, list[list[float]]]]:
    """Return list of (image_basename, rows) where each row is list of floats."""
    blocks: list[tuple[str, list[list[float]]]] = []
    current_name: str | None = None
    current_rows: list[list[float]] = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Image Prediction:"):
                if current_name is not None:
                    blocks.append((current_name, current_rows))
                current_name = line.split(":", 1)[1].strip()
                current_rows = []
                continue
            if line.startswith("Boxes"):
                continue
            parts = line.split()
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if len(nums) == 5:
                current_rows.append(nums + [0.0])
            elif len(nums) >= 6:
                current_rows.append(nums[:6])
            else:
                continue

    if current_name is not None:
        blocks.append((current_name, current_rows))

    return blocks


def _draw_image(
    image_name: str,
    rows: list[list[float]],
    image_folder: str,
    save_folder: str,
    img_width: int,
    img_height: int,
    gt_labels: dict[int, str] | None,
) -> None:
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    h0, w0 = img.shape[:2]
    sx = w0 / float(img_width)
    sy = h0 / float(img_height)

    for row in rows:
        x, y, w, h, conf = row[:5]
        yolo_cls = int(row[5]) if len(row) > 5 else 0
        x = int(float(x) * sx)
        y = int(float(y) * sy)
        w = int(float(w) * sx)
        h = int(float(h) * sy)
        color = _class_bgr(yolo_cls)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        name = _label_for_class(yolo_cls, gt_labels)
        text = f"{name} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x, max(0, y - th - 4)), (x + tw + 2, y), color, -1)
        cv2.putText(
            img,
            text,
            (x + 1, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = os.path.join(save_folder, image_name)
    os.makedirs(save_folder, exist_ok=True)
    cv2.imwrite(out_path, img)


def visualize_boxes(
    pred_file: str,
    image_folder: str = "test_data",
    save_folder: str = "output_visualized",
    img_height: int = 416,
    img_width: int = 416,
    gt_path: str | None = None,
) -> None:
    gt_labels = _load_class_labels(gt_path)
    blocks = _parse_pred_file(pred_file)
    os.makedirs(save_folder, exist_ok=True)

    for image_name, rows in blocks:
        if not rows:
            continue
        _draw_image(image_name, rows, image_folder, save_folder, img_width, img_height, gt_labels)

    print(f"Saved visualizations to {os.path.abspath(save_folder)} ({len(blocks)} image(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize post_processing text output")
    parser.add_argument("pred_file", help="Path to output_boxes_<model>.txt")
    parser.add_argument("img_height", type=int, help="Input height used in post_processing")
    parser.add_argument("img_width", type=int, help="Input width used in post_processing")
    parser.add_argument("--images", default="test_data", help="Folder containing input images")
    parser.add_argument("--out", default="output_visualized", help="Output folder for drawn images")
    parser.add_argument(
        "--gt",
        default=None,
        help="Optional gt_eval.json for class name hints on labels",
    )
    args = parser.parse_args()

    gt = args.gt
    if gt and not os.path.isabs(gt):
        gt = os.path.join(os.getcwd(), gt)

    visualize_boxes(
        args.pred_file,
        image_folder=args.images,
        save_folder=args.out,
        img_height=args.img_height,
        img_width=args.img_width,
        gt_path=gt,
    )


if __name__ == "__main__":
    main()
