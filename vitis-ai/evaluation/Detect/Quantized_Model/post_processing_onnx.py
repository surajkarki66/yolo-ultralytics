import argparse
import onnxruntime as ort
import pickle
import torch
import numpy as np
import re
import os
import time

from glob import glob
from PIL import Image
from utils import non_max_suppression


def apply_dfl(box_preds, reg_max=16):
    """
    Apply Distribution Focal Loss operation.
    
    Args:
        box_preds: Tensor of shape [batch, channels, anchors]
        reg_max: DFL channels
    
    Returns:
        Tensor of shape [batch, 4, anchors] with box coordinates
    """
    b, _, a = box_preds.shape  # batch, channels, anchors
    # Reshape to [b, 4, reg_max, a]
    box_preds = box_preds.view(b, 4, reg_max, a)
    # Transpose to [b, 4, a, reg_max] for softmax
    box_preds = box_preds.transpose(2, 3)
    # Apply softmax over reg_max dimension
    box_preds = box_preds.softmax(3)
    # Create weight tensor [0, 1, 2, ..., reg_max-1]
    weights = torch.arange(reg_max, dtype=box_preds.dtype, device=box_preds.device)
    weights = weights.view(1, 1, 1, reg_max)
    # Weighted sum over the last dimension
    box_preds = (box_preds * weights).sum(3)  # [b, 4, a]
    return box_preds


def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        print(f"WARNING !! failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0

def check_version(current: str = "0.0.0", required: str = "0.0.0", name: str = "version", hard: bool = False, verbose: bool = False, msg: str = "") -> bool:
    from importlib import metadata
    if not current:
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError((f"WARNING !! {current} package is required but not installed")) from e
            else:
                return False

    if not required:
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()
        v = parse_version(version)
        if op == "==" and c != v: result = False
        elif op == "!=" and c == v: result = False
        elif op in {">=", ""} and not (c >= v): result = False
        elif op == "<=" and not (c <= v): result = False
        elif op == ">" and not (c > v): result = False
        elif op == "<" and not (c < v): result = False
    if not result:
        warning = f"WARNING !! {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard: raise ModuleNotFoundError(warning)
        if verbose: print(warning)
    return result

TORCH_1_10 = check_version(torch.__version__, "1.10.0")

def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def decode_bboxes(bboxes, anchors):
    return dist2bbox(bboxes, anchors, xywh=True, dim=1)

def _coerce_strides(tensor_stride) -> list[int]:
    if hasattr(tensor_stride, "tolist"):
        tensor_stride = tensor_stride.tolist()
    return [int(s) for s in tensor_stride]


def load_detect_config(pkl_path: str) -> dict:
    """
    Load Vitis quantization config from vai_q_yolo.py:
    (tensor_no, tensor_stride, tensor_reg_max, tensor_nc, layer_dfl).
    """
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Config pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        cfg = pickle.load(f)
    if not isinstance(cfg, (list, tuple)) or len(cfg) < 4:
        raise ValueError(
            f"Expected pickle tuple (tensor_no, tensor_stride, tensor_reg_max, tensor_nc, ...), "
            f"got {type(cfg)}"
        )
    tensor_no, tensor_stride, tensor_reg_max, tensor_nc = cfg[0], cfg[1], cfg[2], cfg[3]
    return {
        "tensor_no": int(tensor_no),
        "tensor_stride": _coerce_strides(tensor_stride),
        "tensor_reg_max": int(tensor_reg_max),
        "tensor_nc": int(tensor_nc),
    }


def _infer_nc_from_channels(tensor_no: int, reg_max: int) -> int:
    nc = tensor_no - 4 * reg_max
    if nc < 1:
        raise ValueError(
            f"Cannot infer class count: tensor channels={tensor_no}, reg_max={reg_max} "
            f"implies nc={nc}. Try --reg-max or --nc."
        )
    return nc


def run_model_config(x, strides: list[int], reg_max: int = 16, nc: int | None = None):
    """
    Decode raw ONNX feature maps to [batch, 4 + nc, num_anchors] (xywh + class logits).

    If nc is None, it is inferred as: channels_first_output - 4 * reg_max.
    """
    shape = x[0].shape
    tensor_no = int(shape[1])
    tensor_stride = list(strides)

    if nc is None:
        tensor_nc = _infer_nc_from_channels(tensor_no, reg_max)
    else:
        tensor_nc = int(nc)
        expected = 4 * reg_max + tensor_nc
        if tensor_no != expected:
            raise ValueError(
                f"--nc={tensor_nc} and --reg-max={reg_max} expect {expected} channels per scale, "
                f"but ONNX output has {tensor_no}."
            )

    x_cat = torch.cat([xi.view(shape[0], tensor_no, -1) for xi in x], 2)
    tensor_anchors, tensor_strides = (x.transpose(0, 1) for x in make_anchors(x, tensor_stride, 0.5))
    box, cls = x_cat.split((reg_max * 4, tensor_nc), 1)
    dbox = decode_bboxes(apply_dfl(box, reg_max), tensor_anchors.unsqueeze(0)) * tensor_strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y


def res_exp(pred, strides, iou, writer, reg_max: int = 16, nc: int | None = None):
    start = time.time()

    predi = run_model_config(pred, strides, reg_max=reg_max, nc=nc)
    nc_nms = predi.shape[1] - 4
    box_conf = non_max_suppression(
        prediction=predi, conf_thres=0.001, iou_thres=iou, nc=nc_nms
    )
    box_conf[0] = box_conf[0].detach().numpy()
    box_conf[0][box_conf[0] < 0] = 0

    # Keep class_id (col 5): output format is [x, y, w, h, conf, class_id]
    fixed = [
        [i[0], i[1], i[2] - i[0], i[3] - i[1], i[4], int(i[5])]
        for i in box_conf[0]
    ]

    end = time.time() - start
    writer.write("Boxes (x, y, w, h, conf, class_id):\n\n")
    writer.write("\n".join(" ".join(map(str, x)) for x in np.array(fixed)))
    writer.write("\n\n")
    return end


def box_processing_writing(
    dictionary, strides, iou, writer, reg_max: int = 16, nc: int | None = None
):
    total_time = 0.0
    for named, tensor in zip(dictionary["names"], dictionary["tensors"]):
        writer.write(f"Image Prediction: {named}\n\n")
        total_time += res_exp(tensor, strides, iou, writer, reg_max=reg_max, nc=nc)
    return total_time

def preprocess_image(img_path, size=(256, 256)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    paths = []
    for f in sorted(glob(os.path.join(folder, "*"))):
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in _IMG_EXT:
            paths.append(f)
    return paths


def tensor_fix(onnx_path: str, images_dir: str, img_size=(416, 416)):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    tensor_index = {"names": [], "tensors": []}

    images_dir = os.path.abspath(os.path.normpath(images_dir))
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"--images-dir does not exist or is not a directory: {images_dir}")
    image_paths = _list_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images ({', '.join(sorted(_IMG_EXT))}) found in {images_dir}"
        )

    for img_path in image_paths:
        img_input = preprocess_image(img_path, size=img_size)
        outputs = session.run(output_names, {input_name: img_input})
        tensor_index["names"].append(os.path.basename(img_path))
        tensor_index["tensors"].append(tuple(torch.from_numpy(out) for out in outputs))

    return tensor_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX post-processing for YOLOv8 detect")
    parser.add_argument("--onnx", required=True, help="Path to the quantized .onnx model")
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to Vitis config .pkl (tensor_no, stride, reg_max, nc, ... from quantization)",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        metavar="DIR",
        help="Directory containing input images (flat folder; no fixed subfolder name)",
    )
    parser.add_argument("--iou", type=float, required=True, help="NMS IoU threshold")
    parser.add_argument("--img-height", type=int, required=True)
    parser.add_argument("--img-width", type=int, required=True)
    parser.add_argument(
        "--reg-max",
        type=int,
        default=16,
        help="DFL bins (default: from .pkl tensor_reg_max)",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="Number of classes (default: from .pkl; may still verify against ONNX channels)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output text path (default: output_boxes_<onnx_stem>.txt in cwd)",
    )
    args = parser.parse_args()

    cfg = load_detect_config(args.pkl)
    reg_max = args.reg_max if args.reg_max is not None else cfg["tensor_reg_max"]
    nc = args.nc if args.nc is not None else cfg["tensor_nc"]

    img_size = (args.img_width, args.img_height)
    onnx_stem = os.path.splitext(os.path.basename(args.onnx))[0]
    out_path = args.output or f"output_boxes_{onnx_stem}.txt"

    x_dict = tensor_fix(args.onnx, args.images_dir, img_size=img_size)
    with open(out_path, "w") as writer:
        timing = box_processing_writing(
            x_dict,
            cfg["tensor_stride"],
            args.iou,
            writer,
            reg_max=reg_max,
            nc=nc,
        )
    image_count = len(x_dict["names"])
    avg = timing / image_count if image_count else 0.0
    if x_dict["tensors"]:
        ch = int(x_dict["tensors"][0][0].shape[1])
        print(f"nc={nc} (output channels={ch}, reg_max={reg_max})")
    print(f"Total time for image post-processing: {timing}s, Average per image: {avg}s")
