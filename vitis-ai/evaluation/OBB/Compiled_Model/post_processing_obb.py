import math
import os
import pickle

import torch

from utils.obb_utils import dist2rbox
from utils.util import make_anchors


def apply_dfl(box_preds, reg_max=16):
    """DFL: box_preds (b, 4*reg_max, n) -> (b, 4, n)."""
    b, _, a = box_preds.shape
    box_preds = box_preds.view(b, 4, reg_max, a).permute(0, 1, 3, 2)  # (b, 4, a, reg_max)
    box_preds = box_preds.softmax(-1)
    weights = torch.arange(reg_max, dtype=box_preds.dtype, device=box_preds.device)
    box_preds = (box_preds * weights).sum(-1)  # (b, 4, n)
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


def run_obb_post_process(feats, angle_list, config_path, nc=None):
    """
    feats: list of 3 tensors (B, tensor_no, Hi, Wi) = box+cls raw.
    angle_list: list of 3 tensors (B, 1, Hi, Wi) = angle raw.
    Returns: (B, N, 4+nc+1) [xywh, cls_scores..., angle] in grid*stride space.

    nc: if set, overrides class count from the config pickle (must satisfy tensor_ch*4 + nc == C).
    """
    tensor_no_cfg, tensor_stride, tensor_ch, tensor_nc_pkl = load_model_config(config_path)
    device = feats[0].device
    dtype = feats[0].dtype
    B = feats[0].shape[0]
    tensor_no = int(feats[0].shape[1])
    if tensor_no != int(tensor_no_cfg):
        raise ValueError(
            f"Feature map channels {tensor_no} != config tensor_no {tensor_no_cfg}"
        )
    tensor_ch = int(tensor_ch)
    tensor_nc = int(nc) if nc is not None else int(tensor_nc_pkl)
    box_ch = tensor_ch * 4
    if box_ch + tensor_nc != tensor_no:
        raise ValueError(
            f"Channel split invalid: tensor_ch*4 + nc = {box_ch} + {tensor_nc} != {tensor_no} "
            f"(pkl tensor_nc={tensor_nc_pkl}). Adjust --nc or fix the config pickle."
        )
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
    box_raw, cls_logits = x_cat.split((box_ch, tensor_nc), dim=1)

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
