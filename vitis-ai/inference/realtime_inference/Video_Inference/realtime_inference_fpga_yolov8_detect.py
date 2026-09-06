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


# FPGA / VART helpers
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


# YOLOv8 decode helpers
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
    return np.stack((sx, sy), axis=-1).reshape(-1, 2)  # (h*w, 2)


def class_aware_nms_boxes(
    xyxy: np.ndarray,
    scores: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int = 300,
) -> np.ndarray:
    """
    Simple class-aware NMS using `cv2.dnn.NMSBoxes`.

    Returns:
      det: (k, 6) => x1,y1,x2,y2,conf,cls
    """
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1).astype(np.float32)
    keep = conf >= conf_threshold
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    boxes = xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    conf_list = conf.tolist()

    max_wh = 7680.0
    boxes_for_nms = np.stack(
        [x1 + cls * max_wh, y1 + cls * max_wh, w, h], axis=1
    ).tolist()

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_for_nms,
        scores=conf_list,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )
    if len(idxs) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    keep_idx = idxs.flatten()[:max_det]
    det = np.concatenate(
        [boxes[keep_idx], conf[keep_idx, None], cls[keep_idx, None]],
        axis=1,
    )
    return det.astype(np.float32)


def decode_from_fpga(
    out_dequant: List[np.ndarray],
    reorder_out_indices: List[int],
    transpose_needed_flags: List[bool],
    level_anchor_starts: List[int],
    level_hws: List[Tuple[int, int]],
    dist_logits_cat: np.ndarray,
    cls_logits_cat: np.ndarray,
    anchors_cat: np.ndarray,
    scale4_cat: np.ndarray,
    proj: np.ndarray,
    reg_max: int,
    nc: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast decode with precomputed anchors/stride scaling and output reorder mapping.

    Returns:
      xyxy_in: (anchors_total, 4)
      scores: (anchors_total, nc)
    """
    # Fill concatenated dist/cls tensors from per-level outputs.
    for level_i, out_idx in enumerate(reorder_out_indices):
        start = level_anchor_starts[level_i]
        h, w = level_hws[level_i]
        a = h * w

        out = out_dequant[out_idx]
        if transpose_needed_flags[out_idx]:
            out = out.transpose(0, 3, 1, 2)

        dist_logits = out[:, : 4 * reg_max, :, :].reshape(1, 4, reg_max, a)
        cls_logits = out[:, 4 * reg_max : 4 * reg_max + nc, :, :].reshape(1, nc, a)

        dist_logits_cat[:, :, :, start : start + a] = dist_logits
        cls_logits_cat[:, :, start : start + a] = cls_logits

    if reg_max > 1:
        dist_prob = softmax(dist_logits_cat, axis=2)
        pred_dist = np.sum(dist_prob * proj, axis=2)  # [1,4,anchors]
    else:
        pred_dist = dist_logits_cat.reshape(1, 4, -1)

    lt = np.transpose(pred_dist[:, 0:2, :], (0, 2, 1))  # [1,anchors,2]
    rb = np.transpose(pred_dist[:, 2:4, :], (0, 2, 1))  # [1,anchors,2]

    x1y1 = anchors_cat[None] - lt
    x2y2 = anchors_cat[None] + rb
    xyxy = np.concatenate((x1y1, x2y2), axis=-1)  # [1,anchors,4]
    xyxy = xyxy * scale4_cat[None]

    scores = sigmoid(np.transpose(cls_logits_cat, (0, 2, 1)))  # [1,anchors,nc]
    return xyxy[0].astype(np.float32), scores[0].astype(np.float32)


def letterbox_transform_boxes(xyxy: np.ndarray, meta: dict) -> np.ndarray:
    """
    Transform boxes from letterbox space back to original image coordinates.

    Args:
        xyxy: Boxes in letterbox space (target_width x target_height)
        meta: Metadata from letterbox_preprocess

    Returns:
        Boxes in original image coordinates
    """
    gain = meta["gain"]
    pad_w = meta["pad_w"]
    pad_h = meta["pad_h"]

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_w) / gain
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_h) / gain

    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, meta["w0"])
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, meta["h0"])

    return xyxy


# Realtime loop
def run_realtime_detect(
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
    in_shape = tuple(input_t.dims)  # expected NHWC
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input tensor dims: {in_shape}")
    in_scale = 2 ** input_t.get_attr("fix_point")

    out_tensors = runner.get_output_tensors()
    out_fixpoints = [t.get_attr("fix_point") for t in out_tensors]

    with open(config_path, "rb") as f:
        t_no, t_stride, t_reg_max, t_nc = pickle.load(f)[:4]
    expected_strides = list(t_stride)
    imgsz = int(in_shape[1])
    in_h = int(in_shape[1])
    in_w = int(in_shape[2])
    input_is_nhwc = in_shape[-1] == 3
    input_is_nchw = in_shape[1] == 3 and not input_is_nhwc
    if not (input_is_nhwc or input_is_nchw):
        raise ValueError(f"Unsupported input layout for dims {in_shape}. Expected RGB on channel axis.")
    print(f"Input shape: {in_shape}, imgsz={imgsz}")
    print(f"Preprocessing: Letterbox (aspect-ratio preserving + gray pad 114)")
    print(f"Decoder params: reg_max={t_reg_max}, nc={t_nc}, expected_strides={expected_strides}")

    reg_max = int(t_reg_max)
    nc = int(t_nc)
    min_c = 4 * reg_max + nc

    # Precompute output layout and stride-based reorder mapping.
    out_shapes = [tuple(t.dims) for t in out_tensors]
    if len(out_shapes) != len(expected_strides):
        raise ValueError(
            f"Detect expects {len(expected_strides)} output levels, got {len(out_shapes)}"
        )
    if int(in_shape[0]) != 1:
        raise ValueError(f"This script currently supports batch size 1, got input batch {in_shape[0]}")
    for s in out_shapes:
        if int(s[0]) != 1:
            raise ValueError(f"This script currently supports batch size 1, got output batch {s[0]}")
    # DPU output: NHWC.
    # Decoder expects NCHW, so each output is transposed in decode_from_fpga.
    output_layouts: List[Tuple[bool, int, int]] = []
    transpose_needed_flags: List[bool] = []
    stride_ests: List[float] = []
    for out_shape in out_shapes:
        if len(out_shape) != 4:
            raise ValueError(f"Expected 4D output tensor, got {out_shape}")
        c_last = int(out_shape[3])
        if c_last < min_c:
            raise ValueError(
                f"Hardcoded NHWC expects channels on last axis (>= {min_c}), got shape {out_shape}"
            )
        tn = True
        h = int(out_shape[1])
        w = int(out_shape[2])
        output_layouts.append((tn, h, w))
        transpose_needed_flags.append(tn)
        stride_ests.append(float(imgsz) / float(h))

    remaining = set(range(len(out_shapes)))
    reorder_out_indices: List[int] = []
    level_hws = []
    for s_expected in expected_strides:
        best_idx = None
        best_diff = None
        for idx in remaining:
            diff = abs(stride_ests[idx] - float(s_expected))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx
        assert best_idx is not None
        remaining.remove(best_idx)
        reorder_out_indices.append(best_idx)
        _tn, h, w = output_layouts[best_idx]
        level_hws.append((h, w))

    # Precompute concatenated anchors and stride scaling.
    level_anchor_starts: List[int] = []
    anchors_parts: List[np.ndarray] = []
    stride_parts: List[np.ndarray] = []
    total_anchors = 0
    for level_i, (h, w) in enumerate(level_hws):
        level_anchor_starts.append(total_anchors)
        a = h * w
        anchors_parts.append(make_anchors(h, w))
        stride_parts.append(np.full((a,), float(expected_strides[level_i]), dtype=np.float32))
        total_anchors += a

    anchors_cat = np.concatenate(anchors_parts, axis=0).astype(np.float32)  # (total_anchors,2)
    stride_flat = np.concatenate(stride_parts, axis=0).astype(np.float32)  # (total_anchors,)
    scale4_cat = np.repeat(stride_flat[:, None], 4, axis=1).astype(np.float32)  # (total_anchors,4)

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

    # Reuse allocations
    in_data = [np.empty(in_shape, dtype=np.int8, order="C")]
    out_data = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in out_tensors]
    out_dequant = [np.empty(tuple(t.dims), dtype=np.float32, order="C") for t in out_tensors]
    out_scales = [float(2 ** fp) for fp in out_fixpoints]

    # Decoder workspaces
    dist_logits_cat = np.empty((1, 4, reg_max, total_anchors), dtype=np.float32)
    cls_logits_cat = np.empty((1, nc, total_anchors), dtype=np.float32)

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

        xyxy_in, scores = decode_from_fpga(
            out_dequant=out_dequant,
            reorder_out_indices=reorder_out_indices,
            transpose_needed_flags=transpose_needed_flags,
            level_anchor_starts=level_anchor_starts,
            level_hws=level_hws,
            dist_logits_cat=dist_logits_cat,
            cls_logits_cat=cls_logits_cat,
            anchors_cat=anchors_cat,
            scale4_cat=scale4_cat,
            proj=proj,
            reg_max=reg_max,
            nc=nc,
        )

        xyxy_in = letterbox_transform_boxes(xyxy_in, meta)

        det = class_aware_nms_boxes(
            xyxy=xyxy_in,
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

        for x1, y1, x2, y2, conf, cls_id in det:
            cls_i = int(cls_id)
            label = f"{cls_i} {conf:.2f}"
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                label,
                (int(x1), max(int(y1) - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg"), frame
        )

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
    p = argparse.ArgumentParser(
        description="Realtime FPGA inference for YOLOv8 (Detect head) using VART."
    )
    p.add_argument("model_path", help="Path to .xmodel")
    p.add_argument("config_path", help="Path to config.pkl (decode params + quant info)")
    p.add_argument("video_input", help="Video file or camera index (int)")
    p.add_argument("output_dir", help="Output directory for annotated frames")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_input: Union[str, int]
    video_input = (
        int(args.video_input)
        if str(args.video_input).isdigit()
        else args.video_input
    )
    run_realtime_detect(
        model_path=args.model_path,
        config_path=args.config_path,
        video_input=video_input,
        output_dir=args.output_dir,
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
    )
