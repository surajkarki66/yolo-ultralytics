import argparse
import json
import os
import pickle
import sys
import cv2
import numpy as np
import torch
import yaml

from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_QM = os.path.dirname(os.path.abspath(__file__))
if _QM not in sys.path:
    sys.path.insert(0, _QM)
from post_processing import image_paths_from_yolo_dict

from utils import util
from utils.obb_utils import xywhr2xyxyxyxy, xyxyxyxy2xywhr


def _names_from_data_yaml(data: dict) -> list[str]:
    raw = data.get("names")
    if raw is None:
        return ["object"]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        def _key(k):
            try:
                return int(k)
            except (TypeError, ValueError):
                return str(k)

        keys = sorted(raw.keys(), key=_key)
        return [str(raw[k]) for k in keys]
    return ["object"]


def _corners_to_xywhr(corners):
    """Convert 4 corners (n, 8) to xywhr (n, 5). angle in radians."""
    if hasattr(corners, "numpy"):
        corners = corners.numpy()
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim == 1:
        corners = corners.reshape(1, -1)
    return xyxyxyxy2xywhr(torch.from_numpy(corners)).numpy()


def load_gt_labels(image_paths, input_size, images_dir="images", labels_dir="labels"):
    """
    Load GT for each image.
    Returns list of (cls, box_xywhr) where box is in input_size space (0..input_size).
    """
    gt_list = []
    for imp in image_paths:
        image_path = Path(imp)
        if images_dir in image_path.parts:
            parts = list(image_path.parts)
            parts[parts.index(images_dir)] = labels_dir
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            gt_list.append((np.zeros((0, 1), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)))
            continue
        with open(label_path) as f:
            lines = [line.split() for line in f.read().strip().splitlines() if line.strip()]
        if not lines:
            gt_list.append((np.zeros((0, 1), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)))
            continue
        label = np.array(lines, dtype=np.float32)
        if label.shape[1] >= 9:
            cls = label[:, 0:1]
            corners = label[:, 1:9]
            xywhr = _corners_to_xywhr(corners)
            box = xywhr.astype(np.float32)
        elif label.shape[1] >= 6:
            cls = label[:, 0:1]
            box = label[:, 1:6].astype(np.float32)
        elif label.shape[1] == 5:
            cls = label[:, 0:1]
            box = np.concatenate([label[:, 1:5], np.zeros((len(label), 1), dtype=np.float32)], axis=1)
        else:
            cls = np.zeros((0, 1), dtype=np.float32)
            box = np.zeros((0, 5), dtype=np.float32)
        # Scale to input_size
        box = box.copy()
        box[:, :4] *= input_size
        gt_list.append((cls, box))
    return gt_list


def _draw_gt_box(im, xywhr, color=(0, 255, 0), label="GT"):
    """Draw one OBB (xywhr) on im with given color. xywhr: (5,) array."""
    t = torch.as_tensor(xywhr[:5], dtype=torch.float32).unsqueeze(0)
    corners = xywhr2xyxyxyxy(t).squeeze(0).cpu().numpy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(im, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        x1, y1 = int(corners[:, 0].min()), int(corners[:, 1].min())
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(im, label, (x1, max(y1 - 2, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def save_annotated_images(image_paths, pred_list, gt_list, names, input_size, save_dir):
    annotated_dir = os.path.join(save_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    for imp, pred_np, (cls_gt, box_gt) in zip(image_paths, pred_list, gt_list):
        im = cv2.imread(imp)
        if im is None:
            continue
        im = cv2.resize(im, (input_size, input_size))
        if box_gt.size > 0:
            for i in range(len(box_gt)):
                _draw_gt_box(im, box_gt[i], color=(0, 255, 0), label=names[int(cls_gt[i, 0])] if int(cls_gt[i, 0]) < len(names) else "GT")
        for i, pred in enumerate(pred_np):
            xywhr = [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]), float(pred[6])]
            cls_id = int(pred[5])
            conf = float(pred[4])
            label = f"{names[cls_id] if cls_id < len(names) else cls_id} {conf:.2f}"
            util.draw_rotated_box(im, xywhr, cls_id, label)
        out_path = os.path.join(annotated_dir, os.path.basename(imp))
        cv2.imwrite(out_path, im)
    print(f"Annotated images saved to: {annotated_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 OBB (no pycocotools)")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Same data.yaml as post_processing.py (include `names:` for plot/overlay labels)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="PKL from post_processing.py (image_names + predictions)",
    )
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--save-dir", type=str, default=None, help="Save metrics JSON and plots here")
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save test images with GT (green) and predictions drawn",
    )
    args = parser.parse_args()

    data_yaml = os.path.abspath(args.data)
    if not os.path.isfile(data_yaml):
        print(f"data.yaml not found: {data_yaml}")
        sys.exit(1)
    yaml_dir = os.path.dirname(data_yaml)
    with open(data_yaml, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    if not isinstance(yaml_data, dict):
        print(f"Invalid data.yaml (expected mapping): {data_yaml}")
        sys.exit(1)
    names = _names_from_data_yaml(yaml_data)
    _, image_paths = image_paths_from_yolo_dict(yaml_data, yaml_dir)
    image_paths = sorted(image_paths)
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    if not image_paths:
        print("No images found for data.yaml test: split.")
        sys.exit(1)
    print(f"Evaluating on {len(image_paths)} images.")

    gt_list = load_gt_labels(image_paths, args.input_size)
    basenames = [os.path.basename(p) for p in image_paths]

    pred_path = os.path.abspath(args.predictions)
    if not os.path.isfile(pred_path):
        print(f"Predictions file not found: {pred_path}")
        sys.exit(1)
    with open(pred_path, "rb") as f:
        data = pickle.load(f)
    pred_names = data["image_names"]
    predictions = data["predictions"]
    name_to_pred = dict(zip(pred_names, predictions))
    pred_list = [name_to_pred.get(b, np.zeros((0, 7))) for b in basenames]

    # Metrics
    iou_v = torch.linspace(0.5, 0.95, 10)
    metrics = []
    for pred_np, (cls_gt, box_gt) in zip(pred_list, gt_list):
        pred_t = torch.from_numpy(pred_np).float()
        cls_gt_t = torch.from_numpy(cls_gt).float().squeeze(-1)
        box_gt_t = torch.from_numpy(box_gt).float()
        # GT box is already in input_size space
        target_xywhr = box_gt_t
        if pred_t.shape[0] == 0:
            metrics.append((
                torch.zeros(0, iou_v.numel(), dtype=torch.bool),
                torch.zeros(0),
                torch.zeros(0),
                cls_gt_t,
            ))
            continue
        if cls_gt_t.numel() == 0:
            metric = torch.zeros(pred_t.shape[0], iou_v.numel(), dtype=torch.bool)
        else:
            metric = util.compute_metric_obb(pred_t, cls_gt_t, target_xywhr, iou_v)
        metrics.append((metric, pred_t[:, 4], pred_t[:, 5], cls_gt_t))

    # Concatenate and compute AP
    tp = torch.cat([m[0] for m in metrics], dim=0).numpy()
    conf = torch.cat([m[1] for m in metrics], dim=0).numpy()
    pred_cls = torch.cat([m[2] for m in metrics], dim=0).numpy()
    target_cls = torch.cat([m[3] for m in metrics], dim=0).numpy()
    if tp.size == 0:
        m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0
        print("No predictions or targets; metrics set to 0.")
    else:
        save_dir = args.save_dir or os.path.join(_REPO_ROOT, "runs", "eval_obb")
        os.makedirs(save_dir, exist_ok=True)
        _, _, m_pre, m_rec, map50, mean_ap = util.compute_ap(
            "n", tp, conf, pred_cls, target_cls, plot=True, names=names, save_dir=save_dir
        )
    print(("%10s" + "%10.3g" * 4) % ("", m_pre, m_rec, map50, mean_ap))

    out = {
        "precision": float(m_pre),
        "recall": float(m_rec),
        "mAP50": float(map50),
        "mAP50_95": float(mean_ap),
    }
    save_dir = args.save_dir or os.path.join(_REPO_ROOT, "runs", "eval_obb")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics and plots saved to: {save_dir}")
    print(f"metrics.json")

    if args.save_annotated:
        save_annotated_images(image_paths, pred_list, gt_list, names, args.input_size, save_dir)


if __name__ == "__main__":
    main()
