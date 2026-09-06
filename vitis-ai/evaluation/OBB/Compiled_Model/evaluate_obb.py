import argparse
import json
import os
import sys
import cv2
import numpy as np
import torch
import yaml

from pathlib import Path

from utils import util
from utils.obb_utils import xyxyxyxy2xywhr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_data_yaml_path(data_dir: str) -> tuple[str, str]:
    """Return (yaml_path, yaml_dir) for data.yaml / dataset.yaml."""
    p = os.path.abspath(os.path.normpath(data_dir))
    if os.path.isfile(p) and p.lower().endswith((".yaml", ".yml")):
        return p, os.path.dirname(p)
    if os.path.isdir(p):
        for name in ("data.yaml", "dataset.yaml"):
            cand = os.path.join(p, name)
            if os.path.isfile(cand):
                return cand, p
        raise FileNotFoundError(
            f"No data.yaml or dataset.yaml in directory: {p}"
        )
    raise FileNotFoundError(f"Not a YAML file or directory: {data_dir!r}")


def _dataset_root(data: dict, yaml_dir: str) -> str:
    raw = data.get("path")
    if raw is None or str(raw).strip() == "":
        return yaml_dir
    raw = str(raw).strip()
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(yaml_dir, raw))


def _split_entry_path(data: dict, dataset_root: str, split: str) -> str:
    if split not in data:
        raise ValueError(
            f"data.yaml must define `{split}:` (folder or .txt image list)."
        )
    v = data[split]
    if isinstance(v, list):
        if not v:
            raise ValueError(f"data.yaml `{split}:` is an empty list")
        v = v[0]
    v = str(v).strip()
    if os.path.isabs(v):
        return os.path.normpath(v)
    return os.path.normpath(os.path.join(dataset_root, v))


def _paths_from_txt(txt_path: str) -> list[str]:
    txt_path = os.path.abspath(txt_path)
    base = os.path.dirname(txt_path)
    out: list[str] = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("#", 1)[0].strip()
            if not line:
                continue
            p = line
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(base, p))
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in _IMG_EXT:
                out.append(p)
    return out


def _paths_from_dir(root: str) -> list[str]:
    root_p = Path(root)
    if not root_p.is_dir():
        return []
    return sorted(
        str(p)
        for p in root_p.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_EXT
    )


def image_paths_from_yolo_dict(data: dict, yaml_dir: str, split: str) -> tuple[str, list[str]]:
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping for dataset YAML, got {type(data)}")
    dataset_root = _dataset_root(data, yaml_dir)
    split_path = _split_entry_path(data, dataset_root, split)
    if os.path.isfile(split_path) and split_path.lower().endswith(".txt"):
        paths = _paths_from_txt(split_path)
        label = f"{split} list {split_path}"
    elif os.path.isdir(split_path):
        paths = _paths_from_dir(split_path)
        label = split_path
    else:
        raise FileNotFoundError(
            f"`{split}:` resolves to {split_path!r} — not a directory or .txt list. "
            f"Check path / dataset root ({dataset_root!r})."
        )
    if not paths:
        raise FileNotFoundError(
            f"No images ({', '.join(sorted(_IMG_EXT))}) found for `{split}:` at {split_path}"
        )
    return label, paths


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
        image_path = Path(os.path.normpath(imp))
        if images_dir in image_path.parts:
            parts = list(image_path.parts)
            parts[parts.index(images_dir)] = labels_dir
            label_path = Path(*parts).with_suffix(".txt")
        else:
            label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            gt_list.append((np.zeros((0, 1), dtype=np.float32), np.zeros((0, 5), dtype=np.float32)))
            continue
        with open(str(label_path)) as f:
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


def load_predictions_from_npz(
    npz_path, config_path, iou_threshold, num_detect=3, num_angle=3, nc=None
):
    """
    Load FPGA .npz predictions, dequantize using fix_points, run OBB post-process and NMS.
    NPZ must contain: image_names, output_fixpoints, num_outputs, and pred_{img_idx}_output_{out_idx}.
    Expects 6 outputs: 0,1,2 = detect (box+cls), 3,4,5 = angle (YOLOv8-OBB).
    Returns dict: image_basename -> np.ndarray (k, 7) [x, y, w, h, conf, cls, angle].
    nc: optional override for class count (see post_processing_obb.run_obb_post_process).
    """
    from post_processing_obb import run_obb_post_process
    from utils.util import non_max_suppression_obb

    data = np.load(npz_path)
    image_names = data["image_names"]
    if hasattr(image_names, "tolist"):
        image_names = image_names.tolist()
    else:
        image_names = list(image_names)
    # Handle numpy array of strings
    if image_names and hasattr(image_names[0], "item"):
        image_names = [n.item() if hasattr(n, "item") else str(n) for n in image_names]
    else:
        image_names = [str(n) for n in image_names]

    output_fixpoints = data["output_fixpoints"]
    output_shapes = data.get("output_shapes")
    num_outputs = int(data["num_outputs"])
    if num_outputs != num_detect + num_angle:
        raise ValueError(
            f"NPZ num_outputs={num_outputs} expected {num_detect + num_angle} (3 detect + 3 angle) for OBB"
        )

    # FPGA .xmodel may emit angle first then detect (0,1,2=angle, 3,4,5=detect). Detect from shape.
    detect_first = True
    if output_shapes is not None and len(output_shapes) >= 1:
        # NHWC: last dim is C. If output 0 has C=1 it's angle; if C>1 it's detect.
        s0 = output_shapes[0]
        s0 = tuple(int(x) for x in (s0.tolist() if hasattr(s0, "tolist") else s0))
        if len(s0) >= 4 and s0[-1] == 1:
            detect_first = False  # 0,1,2 = angle; 3,4,5 = detect

    name_to_pred = {}
    for img_idx in range(len(image_names)):
        img_name = image_names[img_idx]
        basename = os.path.basename(img_name) if os.path.sep in str(img_name) else img_name

        pred_outputs = []
        for out_idx in range(num_outputs):
            key = f"pred_{img_idx}_output_{out_idx}"
            pred_int8 = data[key]
            fix_point = int(output_fixpoints[out_idx])
            scale = 2 ** fix_point
            pred_float32 = pred_int8.astype(np.float32) / scale
            # NPZ from FPGA is NHWC (1, H, W, C); post_obb expects NCHW (1, C, H, W)
            if pred_float32.ndim == 4:
                # (1, H, W, C) -> (1, C, H, W)
                pred_float32 = np.transpose(pred_float32, (0, 3, 1, 2))
            pred_outputs.append(torch.from_numpy(pred_float32))

        if detect_first:
            feats = pred_outputs[:num_detect]
            angle_list = pred_outputs[num_detect : num_detect + num_angle]
        else:
            # 0,1,2 = angle, 3,4,5 = detect
            angle_list = pred_outputs[:num_angle]
            feats = pred_outputs[num_angle : num_angle + num_detect]
        outputs_t = run_obb_post_process(feats, angle_list, config_path, nc=nc)
        dets = non_max_suppression_obb(
            outputs_t, confidence_threshold=0.001, iou_threshold=iou_threshold
        )
        name_to_pred[basename] = (
            dets[0].cpu().numpy() if dets[0].numel() else np.zeros((0, 7), dtype=np.float32)
        )
    return name_to_pred


def save_annotated_images(image_paths, pred_list, names, input_size, save_dir):
    """Save each image with predictions drawn. Boxes are in input_size space."""
    annotated_dir = os.path.join(save_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    for imp, pred_np in zip(image_paths, pred_list):
        im = cv2.imread(imp)
        if im is None:
            continue
        im = cv2.resize(im, (input_size, input_size))
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
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing data.yaml (or dataset.yaml), or path to that YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="YAML key for images (e.g. val, test): folder or .txt list, resolved via path + split in data.yaml",
    )
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to precomputed _obb_predictions.pkl (image_names + predictions)")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to .npz from fpga_inference.py (OBB); use with --config for post-process")
    parser.add_argument("--config", type=str, default=None, help="Path to model config .pkl (required with --npz)")
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        metavar="N",
        help="With --npz only: override class count for box/cls split (default: tensor_nc in .pkl; require tensor_ch*4 + nc == feature C)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--save-dir", type=str, default=None, help="Save metrics JSON and plots here")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save all test images with predicted OBB annotations drawn")
    args = parser.parse_args()

    try:
        data_yaml, yaml_dir = resolve_data_yaml_path(args.data_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    with open(data_yaml, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    if not isinstance(yaml_data, dict):
        print(f"Invalid dataset YAML (expected mapping): {data_yaml}")
        sys.exit(1)

    try:
        images_label, image_paths = image_paths_from_yolo_dict(
            yaml_data, yaml_dir, args.split
        )
    except (ValueError, FileNotFoundError) as e:
        print(e)
        sys.exit(1)
    image_paths = sorted(image_paths)
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    if not image_paths:
        print(f"No images found for split {args.split!r}.")
        sys.exit(1)
    print(f"split={args.split!r} -> {images_label} ({len(image_paths)} images)")
    print(f"Evaluating on {len(image_paths)} images.")

    names = _names_from_data_yaml(yaml_data)

    # Load GT in input_size space
    gt_list = load_gt_labels(image_paths, args.input_size)
    basenames = [os.path.basename(p) for p in image_paths]

    # Load or run predictions
    if args.predictions and os.path.isfile(args.predictions):
        import pickle
        with open(args.predictions, "rb") as f:
            data = pickle.load(f)
        pred_names = data["image_names"]
        predictions = data["predictions"]
        name_to_pred = dict(zip(pred_names, predictions))
        pred_list = [name_to_pred.get(b, np.zeros((0, 7))) for b in basenames]
    elif args.npz and os.path.isfile(args.npz):
        if not args.config or not os.path.isfile(args.config):
            print("When using --npz, --config must point to the OBB model config .pkl file.")
            sys.exit(1)
        config_path = os.path.abspath(args.config)
        print(f"Loading NPZ predictions from {args.npz} (fix_point dequantization), config {config_path}")
        name_to_pred = load_predictions_from_npz(
            args.npz,
            config_path,
            args.iou_threshold,
            num_detect=3,
            num_angle=3,
            nc=args.nc,
        )
        pred_list = [name_to_pred.get(b, np.zeros((0, 7))) for b in basenames]
    else:
        print("Provide --predictions (pkl path) or --npz + --config (FPGA NPZ).")
        sys.exit(1)

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
        save_annotated_images(image_paths, pred_list, names, args.input_size, save_dir)


if __name__ == "__main__":
    main()
