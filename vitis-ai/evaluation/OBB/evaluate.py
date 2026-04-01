import argparse
import json
import math
import os
import sys
import cv2
import numpy as np
import torch
import yaml

# Repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import util
from utils.obb_utils import xywhr2xyxyxyxy, xyxyxyxy2xywhr


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
        imp = os.path.normpath(imp)
        # Label path: replace first 'images' with 'labels', extension .txt
        sep = os.path.sep
        if images_dir in imp:
            label_path = imp.replace(images_dir, labels_dir, 1).rsplit(".", 1)[0] + ".txt"
        else:
            label_path = os.path.splitext(imp)[0] + ".txt"
        if not os.path.isfile(label_path):
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


def save_annotated_images(image_paths, pred_list, gt_list, names, input_size, save_dir, include_gt=False):
    """Save each image with predictions (and optionally GT) drawn. Boxes are in input_size space."""
    annotated_dir = os.path.join(save_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    for imp, pred_np, (cls_gt, box_gt) in zip(image_paths, pred_list, gt_list):
        im = cv2.imread(imp)
        if im is None:
            continue
        im = cv2.resize(im, (input_size, input_size))
        if include_gt and box_gt.size > 0:
            for i in range(len(box_gt)):
                _draw_obb_gt(im, box_gt[i], color=(0, 255, 0), label=names[int(cls_gt[i, 0])] if int(cls_gt[i, 0]) < len(names) else "GT")
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
    parser.add_argument("--data-dir", type=str, required=True, help="Data root (contains images/, labels/)")
    parser.add_argument("--val-list", type=str, required=True,
                        help="File listing image paths for evaluation (val or test; one path or basename per line)")
    parser.add_argument("--images-subdir", type=str, default="val2017",
                        help="Subdir under data-dir/images/ when list contains only basenames (e.g. val2017 or test2017)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to _obb_predictions.pkl from post_processing_obb.py")
    parser.add_argument("--onnx", type=str, default=None, help="Path to quantized OBB .onnx (run ONNX if set with --config)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config .pkl (use with --onnx)")
    parser.add_argument("--quantize-dir", type=str, default=None,
                        help="Alternative to --onnx/--config: dir containing <model-name>.onnx and <model-name>_config.pkl")
    parser.add_argument("--model-name", type=str, default="YOLO_int", help="Used only with --quantize-dir")
    parser.add_argument("--iou-threshold", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--hyp", type=str, default=None,
                        help="YAML for params.names; default data/hyps/args.yaml from repo")
    parser.add_argument("--save-dir", type=str, default=None, help="Save metrics JSON and plots here")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save all test images with predicted OBB annotations drawn")
    parser.add_argument("--annotated-include-gt", action="store_true",
                        help="When --save-annotated, also draw ground truth boxes (green)")
    args = parser.parse_args()

    # Resolve list to full paths
    with open(args.val_list) as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    if not lines:
        print("Empty val-list file.")
        sys.exit(1)
    if not lines[0].startswith("/") and "images" not in lines[0] and os.path.sep not in lines[0]:
        image_paths = [os.path.join(args.data_dir, "images", args.images_subdir, os.path.basename(x)) for x in lines]
    else:
        image_paths = [x if os.path.isabs(x) else os.path.join(args.data_dir, x) for x in lines]

    # Filter to existing
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    if not image_paths:
        print("No images found. Check --data-dir and --val-list.")
        sys.exit(1)
    print(f"Evaluating on {len(image_paths)} images.")

    # Class names
    hyp_path = args.hyp or os.path.join(_REPO_ROOT, "data", "hyps", "args.yaml")
    if os.path.isfile(hyp_path):
        with open(hyp_path) as f:
            params = yaml.safe_load(f)
        names = params.get("names", ["object"])
    else:
        names = ["object"]

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
    elif (args.onnx and args.config) or (args.quantize_dir and args.model_name):
        # Run ONNX + post-process on val images
        if args.onnx and args.config:
            onnx_path = os.path.abspath(args.onnx)
            config_path = os.path.abspath(args.config)
        else:
            onnx_path = os.path.join(args.quantize_dir, f"{args.model_name}.onnx")
            config_path = os.path.join(args.quantize_dir, f"{args.model_name}_config.pkl")
        if not os.path.isfile(onnx_path) or not os.path.isfile(config_path):
            print(f"ONNX or config not found: {onnx_path!r}, {config_path!r}")
            sys.exit(1)
        sys.path.insert(0, os.path.dirname(__file__))
        from post_processing_obb import preprocess_image, run_obb_post_process
        from utils.util import non_max_suppression_obb
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        img_size = (args.input_size, args.input_size)
        pred_list = []
        for imp in image_paths:
            img_input = preprocess_image(imp, img_size)
            outputs = session.run(output_names, {input_name: img_input})
            feats = [torch.from_numpy(o).float() for o in outputs[:3]]
            angle_list = [torch.from_numpy(o).float() for o in outputs[3:6]]
            outputs_t = run_obb_post_process(feats, angle_list, config_path)
            dets = non_max_suppression_obb(
                outputs_t, confidence_threshold=0.001, iou_threshold=args.iou_threshold
            )
            pred_list.append(dets[0].cpu().numpy() if dets[0].numel() else np.zeros((0, 7)))
    else:
        print("Provide --predictions (pkl path), or --onnx + --config, or --quantize-dir + --model-name to run ONNX.")
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
        save_annotated_images(
            image_paths, pred_list, gt_list, names, args.input_size, save_dir,
            include_gt=args.annotated_include_gt,
        )


if __name__ == "__main__":
    main()
