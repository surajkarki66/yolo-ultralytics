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

try:
    import yaml
except ImportError as e:
    raise ImportError("Install PyYAML: pip install pyyaml") from e

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.obb_utils import dist2rbox
from utils.util import make_anchors, non_max_suppression_obb


def apply_dfl(box_preds, reg_max=16):
    b, _, a = box_preds.shape
    box_preds = box_preds.view(b, 4, reg_max, a).permute(0, 1, 3, 2)
    box_preds = box_preds.softmax(-1)
    weights = torch.arange(reg_max, dtype=box_preds.dtype, device=box_preds.device)
    box_preds = (box_preds * weights).sum(-1)
    return box_preds


def load_model_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    if isinstance(cfg, (list, tuple)) and len(cfg) >= 4:
        return cfg[0], cfg[1], cfg[2], cfg[3]
    raise ValueError(f"Config must be a sequence with at least 4 elements, got {type(cfg)} len={len(cfg) if isinstance(cfg, (list, tuple)) else 'n/a'}")


def run_obb_post_process(feats, angle_list, config_path):
    tensor_no, tensor_stride, tensor_ch, tensor_nc = load_model_config(config_path)
    device = feats[0].device
    dtype = feats[0].dtype
    B = feats[0].shape[0]
    tensor_stride = [float(s) for s in tensor_stride]
    if isinstance(tensor_stride, (list, tuple)):
        strides = tensor_stride
    else:
        strides = tensor_stride.tolist() if hasattr(tensor_stride, "tolist") else [tensor_stride]

    anchor_points, stride_tensor = make_anchors(feats, strides, 0.5)
    anchor_points = anchor_points.to(device=device, dtype=dtype)
    stride_tensor = stride_tensor.to(device=device, dtype=dtype)

    x_cat = torch.cat([xi.view(B, tensor_no, -1) for xi in feats], dim=2)
    box_raw, cls_logits = x_cat.split((tensor_ch * 4, tensor_nc), dim=1)

    box_dist = apply_dfl(box_raw, tensor_ch)

    pred_angle_raw = torch.cat([a.view(B, 1, -1) for a in angle_list], dim=2)
    pred_angle = (pred_angle_raw.sigmoid() - 0.25) * math.pi

    rbox_xywh = dist2rbox(box_dist, pred_angle, anchor_points, dim=1)
    stride_expand = stride_tensor.T.unsqueeze(0)
    rbox_xywh = rbox_xywh * stride_expand

    rbox_xywh = rbox_xywh.permute(0, 2, 1)
    cls_sigmoid = cls_logits.permute(0, 2, 1).sigmoid()
    angle_cat = pred_angle.permute(0, 2, 1)
    outputs = torch.cat((rbox_xywh, cls_sigmoid, angle_cat), dim=-1)
    return outputs


def preprocess_image(img_path, size=(416, 416)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _dataset_root(data: dict, yaml_dir: str) -> str:
    raw = data.get("path")
    if raw is None or str(raw).strip() == "":
        return yaml_dir
    raw = str(raw).strip()
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(yaml_dir, raw))


def _test_entry_path(data: dict, dataset_root: str) -> str:
    if "test" not in data:
        raise ValueError("data.yaml must define `test:` (folder or .txt image list).")
    v = data["test"]
    if isinstance(v, list):
        if not v:
            raise ValueError("data.yaml `test:` is an empty list")
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


def image_paths_from_yolo_dict(data: dict, yaml_dir: str) -> tuple[str, list[str]]:
    """
    Resolve image list from parsed YOLOv8 data dict using only `test:`.

    `test` may be a directory (images discovered recursively) or a .txt list of paths.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping for dataset YAML, got {type(data)}")

    dataset_root = _dataset_root(data, yaml_dir)
    test_path = _test_entry_path(data, dataset_root)

    if os.path.isfile(test_path) and test_path.lower().endswith(".txt"):
        paths = _paths_from_txt(test_path)
        label = f"test list {test_path}"
    elif os.path.isdir(test_path):
        paths = _paths_from_dir(test_path)
        label = test_path
    else:
        raise FileNotFoundError(
            f"`test:` resolves to {test_path!r} — not a directory or .txt list. "
            f"Check path / dataset root ({dataset_root!r})."
        )

    if not paths:
        raise FileNotFoundError(
            f"No images ({', '.join(sorted(_IMG_EXT))}) found for `test:` at {test_path}"
        )
    return label, paths


def image_paths_from_yolo_data_yaml(yaml_path: str) -> tuple[str, list[str]]:
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    yaml_dir = os.path.dirname(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return image_paths_from_yolo_dict(data, yaml_dir)


def tensor_fix(onnx_path, config_path, image_paths: list[str], img_size=(416, 416)):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not image_paths:
        raise FileNotFoundError("No image paths to process")

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    if len(output_names) != 6:
        raise RuntimeError(f"OBB ONNX must have 6 outputs (3 box+cls, 3 angle), got {len(output_names)}")

    result = {"names": [], "raw_feats": [], "raw_angles": []}
    for img_path in sorted(image_paths):
        img_input = preprocess_image(img_path, size=img_size)
        outputs = session.run(output_names, {input_name: img_input})
        feats = [torch.from_numpy(o).float() for o in outputs[:3]]
        angle_list = [torch.from_numpy(o).float() for o in outputs[3:6]]
        result["names"].append(os.path.basename(img_path))
        result["raw_feats"].append(feats)
        result["raw_angles"].append(angle_list)
    return result, config_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, metavar="YAML", help="data.yaml with path + test:")
    p.add_argument(
        "--output",
        default=None,
        help="Path to the output .pkl file, or an existing directory (writes <onnx_stem>_obb_predictions.pkl there).",
    )
    p.add_argument("--iou-threshold", type=float, default=0.45)
    p.add_argument("--img-size", type=int, nargs=2, default=[416, 416], metavar=("H", "W"))
    args = p.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    config_path = os.path.abspath(args.config)
    data_yaml = os.path.abspath(args.data)
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    yaml_dir = os.path.dirname(data_yaml)
    with open(data_yaml, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    if not isinstance(yaml_data, dict):
        raise ValueError(f"Expected a mapping in {data_yaml}, got {type(yaml_data)}")
    images_label, image_paths = image_paths_from_yolo_dict(yaml_data, yaml_dir)
    img_size = (args.img_size[1], args.img_size[0])

    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    default_pkl_name = f"{stem}_obb_predictions.pkl"
    if args.output:
        out_pkl = os.path.abspath(args.output)
        if os.path.isdir(out_pkl):
            out_pkl = os.path.join(out_pkl, default_pkl_name)
    else:
        out_pkl = os.path.join(os.getcwd(), default_pkl_name)

    print(
        f"ONNX={onnx_path}\nconfig={config_path}\ndata.yaml={data_yaml}\n"
        f"test={images_label} ({len(image_paths)} images)\nout={out_pkl}\n"
        f"iou={args.iou_threshold} size={tuple(args.img_size)}"
    )

    t0 = time.time()
    result, config_path = tensor_fix(onnx_path, config_path, image_paths, img_size)
    all_predictions = []
    for fname, feats, angle_list in zip(result["names"], result["raw_feats"], result["raw_angles"]):
        outputs = run_obb_post_process(feats, angle_list, config_path)
        dets = non_max_suppression_obb(outputs, confidence_threshold=0.001, iou_threshold=args.iou_threshold)
        all_predictions.append(dets[0].cpu().numpy() if dets[0].numel() else np.zeros((0, 7)))

    with open(out_pkl, "wb") as f:
        pickle.dump({"image_names": result["names"], "predictions": all_predictions}, f)

    elapsed = time.time() - t0
    n = len(result["names"])
    print(f"done {n} images {elapsed:.3f}s -> {out_pkl}")


if __name__ == "__main__":
    main()
