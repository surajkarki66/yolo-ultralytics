import argparse
import math
import os
import pickle
import sys
import time
import numpy as np
import onnxruntime as ort
import torch

from pathlib import Path
from PIL import Image

# Repo root for utils
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Utils
from utils.obb_utils import dist2rbox
from utils.util import make_anchors, non_max_suppression_obb


def apply_dfl(box_preds, reg_max=16):
    """DFL: box_preds (b, 4*reg_max, n) -> (b, 4, n)."""
    b, _, a = box_preds.shape
    box_preds = box_preds.view(b, 4, reg_max, a).permute(0, 1, 3, 2)  # (b, 4, a, reg_max)
    box_preds = box_preds.softmax(-1)
    weights = torch.arange(reg_max, dtype=box_preds.dtype, device=box_preds.device)
    box_preds = (box_preds * weights).sum(-1)  # (b, 4, a)
    return box_preds


def load_model_config(config_path):
    """Load config pkl. Returns (tensor_no, tensor_stride, tensor_ch, tensor_nc). Uses first 4 elements only (OBB script)."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    if isinstance(cfg, (list, tuple)) and len(cfg) >= 4:
        return cfg[0], cfg[1], cfg[2], cfg[3]
    raise ValueError(f"Config must be a sequence with at least 4 elements, got {type(cfg)} len={len(cfg) if isinstance(cfg, (list, tuple)) else 'n/a'}")


def run_obb_post_process(feats, angle_list, config_path):
    """
    feats: list of 3 tensors (B, tensor_no, Hi, Wi) = box+cls raw.
    angle_list: list of 3 tensors (B, 1, Hi, Wi) = angle raw.
    Returns: (B, N, 4+nc+1) [xywh, cls_scores..., angle] in grid*stride space.
    """
    tensor_no, tensor_stride, tensor_ch, tensor_nc = load_model_config(config_path)
    device = feats[0].device
    dtype = feats[0].dtype
    B = feats[0].shape[0]
    tensor_stride = [float(s) for s in tensor_stride]
    if isinstance(tensor_stride, (list, tuple)):
        strides = tensor_stride
    else:
        strides = tensor_stride.tolist() if hasattr(tensor_stride, "tolist") else [tensor_stride]

    # Anchors from feats
    anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
    anchor_points = anchor_points.to(device=device, dtype=dtype)
    stride_tensor = stride_tensor.to(device=device, dtype=dtype)

    # Concat box+cls: (B, tensor_no, N)
    x_cat = torch.cat([xi.view(B, tensor_no, -1) for xi in feats], dim=2)
    box_raw, cls_logits = x_cat.split((tensor_ch * 4, tensor_nc), dim=1)

    # DFL -> (B, 4, N)
    box_dist = apply_dfl(box_raw, tensor_ch)

    # Angle: concat and (sigmoid - 0.25) * pi
    pred_angle_raw = torch.cat([a.view(B, 1, -1) for a in angle_list], dim=2)
    pred_angle = (pred_angle_raw.sigmoid() - 0.25) * math.pi  # (B, 1, N)

    # dist2rbox: (B, 4, N) in grid space; anchor_points (N, 2)
    rbox_xywh = dist2rbox(box_dist, pred_angle, anchor_points, dim=1)
    # Scale by stride: (B, 4, N) * (1, N, 1); stride_tensor is (N, 1)
    stride_expand = stride_tensor.T.unsqueeze(0)
    rbox_xywh = rbox_xywh * stride_expand

    # (B, N, 4), (B, N, nc), (B, N, 1)
    rbox_xywh = rbox_xywh.permute(0, 2, 1)
    cls_sigmoid = cls_logits.permute(0, 2, 1).sigmoid()
    angle_cat = pred_angle.permute(0, 2, 1)
    outputs = torch.cat((rbox_xywh, cls_sigmoid, angle_cat), dim=-1)
    return outputs


def preprocess_image(img_path, size=(416, 416)):
    """Resize to size and normalize. Returns (1, 3, H, W) float32."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


def tensor_fix(onnx_path, config_path, test_data_dir, img_size=(416, 416)):
    """Run ONNX (6 outputs) on all images in test_data_dir; return dict of names and raw outputs."""
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isdir(test_data_dir):
        raise FileNotFoundError(f"Test data dir not found: {test_data_dir}")

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    if len(output_names) != 6:
        raise RuntimeError(f"OBB ONNX must have 6 outputs (3 box+cls, 3 angle), got {len(output_names)}")

    exts = {".jpg", ".jpeg", ".png"}
  
    root = Path(test_data_dir)
    image_paths = sorted([str(p) for p in root.rglob("*") if p.suffix.lower() in exts])
    if not image_paths:
        raise FileNotFoundError(f"No images (.jpg/.jpeg/.png) found under {test_data_dir}")

    result = {"names": [], "raw_feats": [], "raw_angles": []}
    for img_path in image_paths:
        img_input = preprocess_image(img_path, size=img_size)
        outputs = session.run(output_names, {input_name: img_input})
        feats = [torch.from_numpy(o).float() for o in outputs[:3]]
        angle_list = [torch.from_numpy(o).float() for o in outputs[3:6]]
        result["names"].append(os.path.basename(img_path))
        result["raw_feats"].append(feats)
        result["raw_angles"].append(angle_list)
    return result, config_path


def main():
    parser = argparse.ArgumentParser(
        description="Post-process YOLOv11 OBB quantized ONNX to rotated boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", required=True, help="Path to quantized OBB .onnx model file")
    parser.add_argument("--config", required=True, help="Path to model config .pkl file")
    parser.add_argument("--test-data", required=True,
                        help="Path to directory containing images to run on (e.g. coco_data_obb or coco_data_obb/images/test2017)")
    parser.add_argument("--split", default=None,
                        help="Use only this subdir (e.g. test2017). Images read from <test-data>/images/<split>/ if present, else <test-data>/<split>/")
    parser.add_argument("--output", default=None, help="Output .pkl path (default: <onnx_stem>_obb_predictions.pkl in cwd)")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, nargs=2, default=[416, 416], metavar=("H", "W"), help="Input height and width")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    config_path = os.path.abspath(args.config)
    test_data_dir = os.path.abspath(args.test_data)
    if args.split:
        # Use only images from this split (e.g. test2017) under coco_data_obb/images/test2017/
        for sub in (os.path.join(test_data_dir, "images", args.split), os.path.join(test_data_dir, args.split)):
            if os.path.isdir(sub):
                test_data_dir = sub
                break
        else:
            print(f"WARNING: --split {args.split} not found under {test_data_dir}/images/ or {test_data_dir}/; using full tree.")
    img_size = (args.img_size[1], args.img_size[0])  # (W, H) for resize

    if args.output:
        out_pkl = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        out_pkl = os.path.join(os.getcwd(), f"{stem}_obb_predictions.pkl")

    print("=" * 60)
    print("YOLOv11 OBB ONNX Post-Processing")
    print("=" * 60)
    print(f"ONNX:    {onnx_path}")
    print(f"Config:  {config_path}")
    print(f"Images:  {test_data_dir}" + (f" (split={args.split})" if args.split else ""))
    print(f"Output:  {out_pkl}")
    print(f"IoU:     {args.iou_threshold}, Size: {args.img_size}")
    print("=" * 60)

    t0 = time.time()
    result, config_path = tensor_fix(onnx_path, config_path, test_data_dir, img_size)
    all_predictions = []
    for fname, feats, angle_list in zip(result["names"], result["raw_feats"], result["raw_angles"]):
        outputs = run_obb_post_process(feats, angle_list, config_path)
        dets = non_max_suppression_obb(outputs, confidence_threshold=0.001, iou_threshold=args.iou_threshold)
        all_predictions.append(dets[0].cpu().numpy() if dets[0].numel() else np.zeros((0, 7)))

    with open(out_pkl, "wb") as f:
        pickle.dump({"image_names": result["names"], "predictions": all_predictions}, f)

    elapsed = time.time() - t0
    n = len(result["names"])
    print(f"Processed {n} images in {elapsed:.3f}s ({elapsed/max(n,1):.3f}s/image)")
    print(f"Predictions saved to: {out_pkl}")


if __name__ == "__main__":
    main()
