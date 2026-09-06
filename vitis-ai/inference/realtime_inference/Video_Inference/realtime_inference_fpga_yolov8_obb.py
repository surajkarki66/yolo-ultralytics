#!/usr/bin/env python3
import argparse
import os
import pickle
import time
import cv2
import numpy as np
import vart
import xir

from typing import List, Tuple, Union


def get_child_subgraph_dpu(graph: "xir.Graph"):
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def letterbox_preprocess(image, target_width, target_height, input_scale):
    """
    Letterbox preprocessing with aspect ratio preservation.
    
    Returns:
        processed: np.int8 array ready for DPU
        meta: dict with transformation parameters
    """
    h0, w0 = image.shape[:2]
    
    # Calculate scale to fit within target dimensions
    r = min(target_width / w0, target_height / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    
    # Resize preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_w = (target_width - new_w) / 2.0
    pad_h = (target_height - new_h) / 2.0
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    
    # Add gray padding
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # BGR -> RGB, normalize, quantize
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0
    padded = (padded * input_scale).astype(np.int8)
    
    meta = {
        "mode": "letterbox",
        "h0": h0,
        "w0": w0,
        "gain": r,
        "pad_w": left,
        "pad_h": top,
        "target_w": target_width,
        "target_h": target_height,
    }
    return padded, meta


def letterbox_transform_obb_boxes(xywha: np.ndarray, meta: dict) -> np.ndarray:
    """
    Transform OBB boxes from letterbox space back to original image coordinates.
    
    Args:
        xywha: Boxes in letterbox space (x, y, w, h, angle) - coordinates in target space
        meta: Metadata from letterbox_preprocess
    
    Returns:
        Boxes in original image coordinates
    """
    gain = meta["gain"]
    pad_w = meta["pad_w"]
    pad_h = meta["pad_h"]
    
    xywha_transformed = xywha.copy()
    
    # Remove padding and scale back for x, y, w, h
    xywha_transformed[:, 0] = (xywha_transformed[:, 0] - pad_w) / gain  # x
    xywha_transformed[:, 1] = (xywha_transformed[:, 1] - pad_h) / gain  # y
    xywha_transformed[:, 2] /= gain  # width
    xywha_transformed[:, 3] /= gain  # height
    
    # Clip to original image boundaries
    xywha_transformed[:, 0] = np.clip(xywha_transformed[:, 0], 0, meta["w0"])
    xywha_transformed[:, 1] = np.clip(xywha_transformed[:, 1], 0, meta["h0"])
    xywha_transformed[:, 2] = np.clip(xywha_transformed[:, 2], 0, meta["w0"])
    xywha_transformed[:, 3] = np.clip(xywha_transformed[:, 3], 0, meta["h0"])
    
    # Angle remains unchanged (rotation is independent of scaling)
    
    return xywha_transformed


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_anchors(h: int, w: int) -> np.ndarray:
    sy, sx = np.meshgrid(
        np.arange(h, dtype=np.float32) + 0.5,
        np.arange(w, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return np.stack((sx, sy), axis=-1).reshape(-1, 2)


def _get_covariance_matrix(boxes: np.ndarray):
    gbbs = np.concatenate([(boxes[..., 2:4] ** 2) / 12.0, boxes[..., 4:5]], axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )


def batch_probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = [x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1)]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = [x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2)]
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom) * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2) / denom) * 0.5
    t3 = np.log(
        ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
        / (4 * np.sqrt(np.clip(a1 * b1 - c1 ** 2, 0, None) * np.clip(a2 * b2 - c2 ** 2, 0, None)) + eps)
        + eps
    ) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd


def rotated_nms_class_aware(
    xywha: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int = 300,
) -> np.ndarray:
    """
    xywha: (N,5) => x,y,w,h,angle(rad)
    scores: (N,nc)
    returns det: (k,7) => x,y,w,h,conf,cls,angle
    """
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1).astype(np.float32)
    keep = conf >= conf_threshold
    if not np.any(keep):
        return np.zeros((0, 7), dtype=np.float32)

    boxes = xywha[keep]
    conf = conf[keep]
    cls = cls[keep]

    order = np.argsort(conf)[::-1]
    boxes = boxes[order]
    conf = conf[order]
    cls = cls[order]

    # class-aware offset on center like YOLOv8 main.py
    max_wh = 7680.0
    boxes_nms = boxes.copy()
    boxes_nms[:, 0] += cls * max_wh
    boxes_nms[:, 1] += cls * max_wh

    ious = batch_probiou(boxes_nms, boxes_nms)
    ious = np.triu(ious, k=1)
    keep_mask = (ious >= iou_threshold).sum(axis=0) <= 0
    keep_idx = np.where(keep_mask)[0][:max_det]
    if keep_idx.size == 0:
        return np.zeros((0, 7), dtype=np.float32)

    picked = np.concatenate(
        [boxes[keep_idx, :4], conf[keep_idx, None], cls[keep_idx, None], boxes[keep_idx, 4:5]],
        axis=1,
    )
    return picked.astype(np.float32)


def draw_rotated_box(im: np.ndarray, det_row: np.ndarray, label: str) -> None:
    x, y, w, h, conf, cls_id, angle = det_row
    rect = ((float(x), float(y)), (max(float(w), 1.0), max(float(h), 1.0)), float(angle * 180.0 / np.pi))
    pts = cv2.boxPoints(rect).astype(np.int32)
    color = (0, 255, 0)
    cv2.polylines(im, [pts], True, color, 2, cv2.LINE_AA)
    cv2.putText(
        im,
        label,
        (int(pts[0][0]), max(int(pts[0][1]) - 6, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def decode_from_fpga(
    out_dequant: List[np.ndarray],
    transpose_needed_flags: List[bool],
    box_indices: List[int],
    angle_indices: List[int],
    level_anchor_starts: List[int],
    level_hws: List[Tuple[int, int]],
    dist_logits_cat: np.ndarray,
    cls_logits_cat: np.ndarray,
    angle_logits_cat: np.ndarray,
    anchors_cat: np.ndarray,
    stride_cat: np.ndarray,
    proj: np.ndarray,
    reg_max: int,
    nc: int,
) -> Tuple[np.ndarray, np.ndarray]:
    for level_i, (bidx, aidx) in enumerate(zip(box_indices, angle_indices)):
        start = level_anchor_starts[level_i]
        h, w = level_hws[level_i]
        a = h * w

        box_out = out_dequant[bidx]
        ang_out = out_dequant[aidx]
        if transpose_needed_flags[bidx]:
            box_out = box_out.transpose(0, 3, 1, 2)
        if transpose_needed_flags[aidx]:
            ang_out = ang_out.transpose(0, 3, 1, 2)

        dist_logits = box_out[:, : 4 * reg_max, :, :].reshape(1, 4, reg_max, a)
        cls_logits = box_out[:, 4 * reg_max : 4 * reg_max + nc, :, :].reshape(1, nc, a)
        ang_logits = ang_out[:, :1, :, :].reshape(1, 1, a)

        dist_logits_cat[:, :, :, start : start + a] = dist_logits
        cls_logits_cat[:, :, start : start + a] = cls_logits
        angle_logits_cat[:, :, start : start + a] = ang_logits

    if reg_max > 1:
        dist_prob = softmax(dist_logits_cat, axis=2)
        pred_dist = np.sum(dist_prob * proj, axis=2)
    else:
        pred_dist = dist_logits_cat.reshape(1, 4, -1)

    # angle decode as in YOLOv8 main.py (raw -> (sigmoid-0.25)*pi or already normalized)
    amin = float(angle_logits_cat.min())
    amax = float(angle_logits_cat.max())
    lo, hi = -0.25 * np.pi, 0.75 * np.pi
    if amin >= (lo - 0.25) and amax <= (hi + 0.25):
        pred_angle = angle_logits_cat
    else:
        pred_angle = (sigmoid(angle_logits_cat) - 0.25) * np.pi

    lt = np.transpose(pred_dist[:, 0:2, :], (0, 2, 1))
    rb = np.transpose(pred_dist[:, 2:4, :], (0, 2, 1))
    ang = np.transpose(pred_angle[:, :1, :], (0, 2, 1))

    xf = (rb[..., 0:1] - lt[..., 0:1]) / 2.0
    yf = (rb[..., 1:2] - lt[..., 1:2]) / 2.0
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    x = xf * cos_a - yf * sin_a
    y = xf * sin_a + yf * cos_a
    xy = np.concatenate((x, y), axis=-1) + anchors_cat[None]
    wh = lt + rb
    xywh = np.concatenate((xy, wh), axis=-1) * stride_cat[None, :, None]

    scores = sigmoid(np.transpose(cls_logits_cat, (0, 2, 1)))
    angles = ang
    xywha = np.concatenate([xywh[0], angles[0]], axis=1).astype(np.float32)
    return xywha, scores[0].astype(np.float32)


def run_realtime_obb(
    model_path: str,
    config_path: str,
    video_input: Union[str, int],
    output_dir: str,
    conf_threshold: float,
    iou_threshold: float,
):
    os.makedirs(output_dir, exist_ok=True)

    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in the xmodel.")
    runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_t = runner.get_input_tensors()[0]
    in_shape = tuple(input_t.dims)
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input tensor dims: {in_shape}")
    in_scale = 2 ** input_t.get_attr("fix_point")
    input_is_nhwc = in_shape[-1] == 3
    input_is_nchw = in_shape[1] == 3 and not input_is_nhwc
    if not (input_is_nhwc or input_is_nchw):
        raise ValueError(f"Unsupported input layout for dims {in_shape}.")

    out_tensors = runner.get_output_tensors()
    out_fixpoints = [t.get_attr("fix_point") for t in out_tensors]
    out_shapes = [tuple(t.dims) for t in out_tensors]
    if len(out_shapes) != 6:
        raise ValueError(f"OBB expects 6 outputs (3 boxcls + 3 angle), got {len(out_shapes)}")

    with open(config_path, "rb") as f:
        _t_no, t_stride, t_reg_max, t_nc = pickle.load(f)[:4]
    expected_strides = list(t_stride)
    reg_max = int(t_reg_max)
    nc = int(t_nc)
    ne = 1

    imgsz = int(in_shape[1])
    in_h = int(in_shape[1])
    in_w = int(in_shape[2])
    print(f"Input shape: {in_shape}, imgsz={imgsz}")
    print("Preprocessing: Letterbox (aspect-ratio preserving + gray pad 114)")
    print(f"Decoder params: reg_max={reg_max}, nc={nc}, ne={ne}, strides={expected_strides}")

    box_ch = 4 * reg_max + nc

    box_idxs: List[int] = []
    angle_idxs: List[int] = []
    for i, s in enumerate(out_shapes):
        c1, cl = int(s[1]), int(s[-1])
        if c1 == box_ch or cl == box_ch:
            box_idxs.append(i)
        if c1 == ne or cl == ne:
            angle_idxs.append(i)
    if len(box_idxs) != 3 or len(angle_idxs) != 3:
        raise ValueError(f"Could not split outputs into 3 boxcls + 3 angle. box_idxs={box_idxs}, angle_idxs={angle_idxs}")

    box_set = set(box_idxs)
    angle_set = set(angle_idxs)
    # DPU layout: NHWC.
    # Decoder expects NCHW, so we always transpose outputs in decode_from_fpga.
    transpose_needed_flags: List[bool] = []
    level_hw_per_out: List[Tuple[int, int]] = []
    for i, s in enumerate(out_shapes):
        if i in box_set:
            mc = box_ch
        elif i in angle_set:
            mc = ne
        else:
            c1, cl = int(s[1]), int(s[-1])
            mc = max(c1, cl)
        if len(s) != 4:
            raise ValueError(f"Expected 4D output tensor, got {s}")
        if int(s[-1]) < mc:
            raise ValueError(
                f"Hardcoded NHWC expects channels on last axis (>= {mc}), got shape {s}"
            )
        tn = True
        h = int(s[1])
        w = int(s[2])
        transpose_needed_flags.append(tn)
        level_hw_per_out.append((h, w))

    def order_by_stride(indices: List[int]) -> List[int]:
        stride_est = {}
        for i in indices:
            h, _w = level_hw_per_out[i]
            stride_est[i] = float(imgsz) / float(h)
        rem = set(indices)
        ordered = []
        for s_exp in expected_strides:
            best_i, best_d = None, None
            for i in rem:
                d = abs(stride_est[i] - float(s_exp))
                if best_d is None or d < best_d:
                    best_i, best_d = i, d
            assert best_i is not None
            rem.remove(best_i)
            ordered.append(best_i)
        return ordered

    box_indices = order_by_stride(box_idxs)
    angle_indices = order_by_stride(angle_idxs)

    level_hws = [level_hw_per_out[i] for i in box_indices]
    level_anchor_starts: List[int] = []
    anchors_parts = []
    stride_parts = []
    total_anchors = 0
    for li, (h, w) in enumerate(level_hws):
        level_anchor_starts.append(total_anchors)
        a = h * w
        anchors_parts.append(make_anchors(h, w))
        stride_parts.append(np.full((a,), float(expected_strides[li]), dtype=np.float32))
        total_anchors += a
    anchors_cat = np.concatenate(anchors_parts, axis=0).astype(np.float32)
    stride_cat = np.concatenate(stride_parts, axis=0).astype(np.float32)
    proj = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_input}")

    frame_idx = 0
    total_fpga_latency_ms = 0.0
    total_end_to_end_latency_ms = 0.0
    
    print("-" * 100)
    print(
        f"{'Frame':<6} | {'FPGA (ms)':<12} | {'FPGA FPS':<10} | {'End-to-End (ms)':<18} | {'End-to-End FPS':<14}"
    )
    print("-" * 100)

    in_data = [np.empty(in_shape, dtype=np.int8, order="C")]
    out_data = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in out_tensors]
    out_dequant = [np.empty(tuple(t.dims), dtype=np.float32, order="C") for t in out_tensors]
    out_scales = [float(2 ** fp) for fp in out_fixpoints]

    dist_logits_cat = np.empty((1, 4, reg_max, total_anchors), dtype=np.float32)
    cls_logits_cat = np.empty((1, nc, total_anchors), dtype=np.float32)
    angle_logits_cat = np.empty((1, 1, total_anchors), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        end_to_end_start = time.perf_counter()

        processed_img, meta = letterbox_preprocess(frame, in_w, in_h, in_scale)
        if input_is_nhwc:
            in_data[0][0, ...] = processed_img
        else:
            in_data[0][0, 0, :, :] = processed_img[:, :, 0]
            in_data[0][0, 1, :, :] = processed_img[:, :, 1]
            in_data[0][0, 2, :, :] = processed_img[:, :, 2]

        fpga_start = time.perf_counter()
        jid = runner.execute_async(in_data, out_data)
        runner.wait(jid)
        fpga_end = time.perf_counter()

        for i, arr in enumerate(out_data):
            np.divide(arr, out_scales[i], out=out_dequant[i])

        xywha_in, scores = decode_from_fpga(
            out_dequant=out_dequant,
            transpose_needed_flags=transpose_needed_flags,
            box_indices=box_indices,
            angle_indices=angle_indices,
            level_anchor_starts=level_anchor_starts,
            level_hws=level_hws,
            dist_logits_cat=dist_logits_cat,
            cls_logits_cat=cls_logits_cat,
            angle_logits_cat=angle_logits_cat,
            anchors_cat=anchors_cat,
            stride_cat=stride_cat,
            proj=proj,
            reg_max=reg_max,
            nc=nc,
        )

        xywha_in = letterbox_transform_obb_boxes(xywha_in, meta)

        det = rotated_nms_class_aware(
            xywha=xywha_in,
            scores=scores,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        end_to_end_end = time.perf_counter()
        
        f_lat = (fpga_end - fpga_start) * 1000.0
        end_to_end_lat = (end_to_end_end - end_to_end_start) * 1000.0
        total_fpga_latency_ms += f_lat
        total_end_to_end_latency_ms += end_to_end_lat
        
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(
                f"{frame_idx:<6} | {f_lat:<12.2f} | {1000.0/f_lat:<10.2f} | {end_to_end_lat:<18.2f} | {1000.0/end_to_end_lat:<10.2f}"
            )

        for row in det:
            cls_i = int(row[5])
            label = f"{cls_i} {float(row[4]):.2f}"
            draw_rotated_box(frame, row, label)

        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), frame)

    cap.release()
    print("-" * 100)
    if frame_idx == 0:
        print("No frames processed.")
        return
    
    avg_fpga = total_fpga_latency_ms / frame_idx
    avg_end_to_end = total_end_to_end_latency_ms / frame_idx
    print(f"AVERAGE FPGA LATENCY: {avg_fpga:.2f} ms ({1000.0/avg_fpga:.2f} FPS)")
    print(f"AVERAGE END-TO-END:   {avg_end_to_end:.2f} ms ({1000.0/avg_end_to_end:.2f} FPS)")
    print(f"OVERHEAD (non-FPGA):  {avg_end_to_end - avg_fpga:.2f} ms per frame")
    print("-" * 100)


def parse_args():
    p = argparse.ArgumentParser(description="Realtime FPGA inference for YOLOv8 OBB using VART.")
    p.add_argument("model_path", help="Path to .xmodel")
    p.add_argument("config_path", help="Path to config.pkl (decode params + quant info)")
    p.add_argument("video_input", help="Video file or camera index (int)")
    p.add_argument("output_dir", help="Output directory for annotated frames")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="Rotated NMS IoU threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_input: Union[str, int] = int(args.video_input) if str(args.video_input).isdigit() else args.video_input
    run_realtime_obb(
        model_path=args.model_path,
        config_path=args.config_path,
        video_input=video_input,
        output_dir=args.output_dir,
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
    )
