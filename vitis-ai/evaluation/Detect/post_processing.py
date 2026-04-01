"""
Post-process ONNX model outputs to bounding boxes and class scores.
Usage: python post_processing.py <path_to_output> <model_name> <iou_threshold> <img_height> <img_width>
"""
import onnxruntime as ort
import torch
import numpy as np
import re
import os
import sys
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

def get_model_config(model_name):
    """
    Get model configuration based on model name.
    
    Returns:
        tuple: (tensor_no, tensor_stride, tensor_reg_max, tensor_nc)
    """
    # Default YOLOv8 configuration
    configs = {
        "default": {
            "tensor_stride": [8, 16, 32],
            "tensor_reg_max": 16,
            "tensor_nc": 1,
        },
    }
    
    config = configs.get(model_name, configs["default"])
    tensor_stride = config["tensor_stride"]
    tensor_reg_max = config["tensor_reg_max"]
    tensor_nc = config["tensor_nc"]
    tensor_no = tensor_reg_max * 4 + tensor_nc
    
    return tensor_no, tensor_stride, tensor_reg_max, tensor_nc


def run_model_config(x, pth, name):
    shape = x[0].shape
    tensor_no, tensor_stride, tensor_reg_max, tensor_nc = get_model_config(name)

    x_cat = torch.cat([xi.view(shape[0], tensor_no, -1) for xi in x], 2)
    tensor_anchors, tensor_strides = (x.transpose(0, 1) for x in make_anchors(x, tensor_stride, 0.5))
    box, cls = x_cat.split((tensor_reg_max * 4, tensor_nc), 1)
    dbox = decode_bboxes(apply_dfl(box, tensor_reg_max), tensor_anchors.unsqueeze(0)) * tensor_strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y

def res_exp(pred, pth, name, iou):
    global f
    start = time.time()

    predi = run_model_config(pred, pth, name)
    box_conf = non_max_suppression(prediction=predi, conf_thres=0.001, iou_thres=iou)
    box_conf[0] = box_conf[0].detach().numpy()
    box_conf[0] = np.delete(box_conf[0], 5, axis=1)
    box_conf[0][box_conf[0] < 0] = 0

    fixed = [[i[0], i[1], i[2]-i[0], i[3]-i[1], i[4]] for i in box_conf[0]]

    end = time.time() - start
    global timing
    timing += end

    f.write("Boxes:\n\n")
    f.write("\n".join(" ".join(map(str, x)) for x in np.array(fixed)))
    f.write("\n\n")

def box_processing_writing(dictionary, pth, name, iou):
    global f
    for named, tensor in zip(dictionary["names"], dictionary["tensors"]):
        f.write(f"Image Prediction: {named}\n\n")
        res_exp(tensor, pth, name, iou)

def preprocess_image(img_path, size=(256, 256)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr

def tensor_fix(pth, model_name, img_size=(416, 416)):
    session = ort.InferenceSession(os.path.join(pth, f"models/{model_name}.onnx"))
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    tensor_index = {"names": [], "tensors": []}

    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    test_folder = os.path.join(pth, "test_data")
    all_files = sorted(glob(os.path.join(test_folder, "*.*")))
    image_paths = [f for f in all_files if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

    for img_path in image_paths:
        img_input = preprocess_image(img_path, size=img_size)
        outputs = session.run(output_names, {input_name: img_input})
        tensor_index["names"].append(os.path.basename(img_path))
        tensor_index["tensors"].append(tuple(torch.from_numpy(out) for out in outputs))

    return tensor_index

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Run as: python post_processing.py <path_to_output> <model_name> <iou_threshold> <img_height> <img_width>")
        sys.exit(1)

    global timing
    timing = 0
    pth = sys.argv[1]
    name = sys.argv[2]
    iou = float(sys.argv[3])
    img_height = int(sys.argv[4])
    img_width = int(sys.argv[5])
    img_size = (img_width, img_height)  

    global f
    f = open(f"output_boxes_{name}.txt", 'w')

    x_dict = tensor_fix(pth, name, img_size=img_size)
    box_processing_writing(x_dict, pth, name, iou)
    print(f'Total time for image post-processing: {timing}s, Average per image: {timing/len(x_dict["names"])}s')