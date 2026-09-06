import argparse
import numpy as np
import pickle
import sys
import os
import time


# ============================================================================
# NUMPY HELPERS
# ============================================================================

def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms_numpy(boxes, scores, iou_threshold):
    """Pure-NumPy Non-Maximum Suppression. Equivalent to torchvision.ops.nms."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) to (x1, y1, x2, y2)."""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    dw = x[..., 2] / 2
    dh = x[..., 3] / 2
    y[..., 0] = x[..., 0] - dw
    y[..., 1] = x[..., 1] - dh
    y[..., 2] = x[..., 0] + dw
    y[..., 3] = x[..., 1] + dh
    return y


def apply_dfl(box_preds, reg_max=16):
    """
    Apply Distribution Focal Loss operation (aligned with YOLOv8/post_processing.py).

    Args:
        box_preds: Array of shape [batch, channels, anchors]
        reg_max: DFL channels

    Returns:
        Array of shape [batch, 4, anchors] with box coordinates
    """
    b, _, a = box_preds.shape                              # [b, reg_max*4, a]
    box_preds = box_preds.reshape(b, 4, reg_max, a)        # [b, 4, reg_max, a]
    box_preds = box_preds.transpose(0, 1, 3, 2)            # [b, 4, a, reg_max]
    box_preds = _softmax(box_preds, axis=-1)
    weights = np.arange(reg_max, dtype=box_preds.dtype).reshape(1, 1, 1, reg_max)
    box_preds = (box_preds * weights).sum(-1)              # [b, 4, a]
    return box_preds


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
):
    """Perform non-maximum suppression (NMS) on a set of boxes."""
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].max(axis=1) > conf_thres      # [bs, anchors]

    time_limit = 2.0 + max_time_img * bs
    multi_label = multi_label and nc > 1

    prediction = prediction.transpose(0, 2, 1)             # [bs, anchors, channels]
    if in_place:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    else:
        prediction = np.concatenate(
            (xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1
        )

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].astype(int) + 4] = 1.0
            x = np.concatenate((x, v), axis=0)

        if x.shape[0] == 0:
            continue

        box  = x[:, :4]
        cls  = x[:, 4:4 + nc]
        mask = x[:, 4 + nc:]

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, np.newaxis], j[:, np.newaxis].astype(float), mask[i]),
                axis=1,
            )
        else:
            j    = cls.argmax(axis=1)
            conf = cls[np.arange(len(cls)), j][:, np.newaxis]
            x    = np.concatenate((box, conf, j[:, np.newaxis].astype(float), mask), axis=1)
            x    = x[conf.ravel() > conf_thres]

        if classes is not None:
            x = x[np.isin(x[:, 5].astype(int), classes)]

        n = x.shape[0]
        if n == 0:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        c      = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes  = x[:, :4] + c
        i      = _nms_numpy(boxes, scores, iou_thres)
        i      = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break

    return output


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = np.arange(w, dtype=dtype) + grid_cell_offset  # [w]
        sy = np.arange(h, dtype=dtype) + grid_cell_offset  # [h]
        # indexing='ij': sy varies along axis-0 (rows), sx along axis-1 (cols)
        sy_grid, sx_grid = np.meshgrid(sy, sx, indexing="ij")  # both [h, w]
        anchor_points.append(np.stack((sx_grid, sy_grid), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=dtype))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=1):
    """Convert distance predictions to bounding box coordinates."""
    lt, rb = np.split(distance, 2, axis=dim)   # each [b, 2, a]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh   = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)
    return np.concatenate((x1y1, x2y2), axis=dim)


def decode_bboxes(bboxes, anchors):
    return dist2bbox(bboxes, anchors, xywh=True, dim=1)


def run_model_config(x, model_config_path, name):
    """Process model outputs with automatic NHWC->NCHW conversion."""
    with open(model_config_path, "rb") as f:
        loaded = pickle.load(f)

    # When len==5 the 5th element is a PyTorch DFL layer module; we discard it
    # and use the equivalent apply_dfl() implemented in NumPy instead.
    tensor_no, tensor_stride, tensor_reg_max, tensor_nc = loaded[:4]

    # Convert inputs; detect NHWC [B, H, W, C] and transpose to NCHW [B, C, H, W]
    x_np = []
    for xi in x:
        arr = np.asarray(xi, dtype=np.float32)
        if arr.ndim == 4:
            B, dim1, dim2, dim3 = arr.shape
            if dim1 > 10 and dim2 > 10 and dim3 < 100:
                arr = arr.transpose(0, 3, 1, 2)         # NHWC -> NCHW
        x_np.append(arr)

    shape = x_np[0].shape                               # [B, C, H, W]
    x_cat = np.concatenate(
        [xi.reshape(shape[0], tensor_no, -1) for xi in x_np], axis=2
    )                                                    # [B, tensor_no, total_anchors]

    anchor_points, stride_tensor = make_anchors(x_np, tensor_stride, 0.5)
    tensor_anchors = anchor_points.T                     # [2, total_anchors]
    tensor_strides = stride_tensor.T                     # [1, total_anchors]

    box = x_cat[:, : tensor_reg_max * 4, :]              # [B, reg_max*4, a]
    cls = x_cat[:, tensor_reg_max * 4 :, :]              # [B, nc, a]

    dbox = (
        decode_bboxes(
            apply_dfl(box, tensor_reg_max),
            tensor_anchors[np.newaxis, :, :],            # [1, 2, a]
        )
        * tensor_strides                                 # [1, a] broadcasts to [B, 4, a]
    )
    y = np.concatenate((dbox, _sigmoid(cls)), axis=1)    # [B, 4+nc, a]
    return y, x_np


def process_detections(pred, model_config_path, name, iou, conf_thres=0.001):
    """Process predictions and return bounding boxes (conf_thres aligned with YOLOv8/post_processing.py)."""
    predi, _ = run_model_config(pred, model_config_path, name)
    box_conf = non_max_suppression(prediction=predi, conf_thres=conf_thres, iou_thres=iou)

    if len(box_conf) > 0 and box_conf[0] is not None and len(box_conf[0]) > 0:
        result = box_conf[0]
        result[result < 0] = 0
        # xyxy -> xywh + conf + yolo_cls (same box text format as ONNX CPU post-process)
        fixed = [
            [i[0], i[1], i[2] - i[0], i[3] - i[1], i[4], int(i[5])]
            for i in result
        ]
    else:
        fixed = []

    return fixed


# ============================================================================
# MAIN POST-PROCESSING FUNCTION
# ============================================================================

def postprocess_predictions(input_npz_path, model_config_path, model_name,
                            iou_threshold, output_file):
    """
    Load predictions from NPZ file and perform post-processing.

    Args:
        input_npz_path   : Path to NPZ file with predictions
        model_config_path: Path to model config pickle file
        model_name       : Model name (e.g., 'yolov8n')
        iou_threshold    : IOU threshold for NMS
        output_file      : Path to save output text file
    """
    print(f"Loading predictions from: {input_npz_path}")

    data = np.load(input_npz_path)

    image_names     = data["image_names"]
    output_shapes   = data["output_shapes"]
    output_fixpoints = data["output_fixpoints"]
    num_outputs     = int(data["num_outputs"])
    num_images      = len(image_names)

    print(f"Loaded predictions for {num_images} images")
    print(f"Number of output tensors per image: {num_outputs}")
    print(f"Output shapes: {output_shapes}")
    print(f"Output fixpoints: {output_fixpoints}")

    f = open(output_file, "w")

    total_postprocess_time = 0
    overall_start_time     = time.time()

    try:
        for img_idx in range(num_images):
            img_name = image_names[img_idx]

            # Dequantize using per-output fix point
            pred_outputs = []
            for out_idx in range(num_outputs):
                key         = f"pred_{img_idx}_output_{out_idx}"
                pred_int8   = data[key]
                fix_point   = int(output_fixpoints[out_idx])
                scale       = 2 ** fix_point
                pred_float32 = pred_int8.astype(np.float32) / scale
                pred_outputs.append(pred_float32)

            postprocess_start = time.time()
            boxes = process_detections(pred_outputs, model_config_path, model_name, iou_threshold)
            postprocess_time   = time.time() - postprocess_start
            total_postprocess_time += postprocess_time

            f.write(f"Image Prediction: {img_name}\n\n")
            f.write("Boxes (x, y, w, h, conf, class_id):\n\n")
            if boxes:
                f.write("\n".join(" ".join(map(str, x)) for x in np.array(boxes)))
            f.write("\n\n")

            if (img_idx + 1) % 10 == 0 or (img_idx + 1) == num_images:
                avg_time = total_postprocess_time / (img_idx + 1)
                print(f"Processed {img_idx + 1}/{num_images} images | "
                      f"Avg: {avg_time * 1000:.2f}ms")

    finally:
        f.close()

    total_processing_time = time.time() - overall_start_time

    print(f"\n{'='*70}")
    print(f"POST-PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total images processed: {num_images}")
    print(f"")
    print(f"📊 POST-PROCESSING:")
    print(f"   Total time: {total_postprocess_time:.2f}s")
    print(f"   Average per image: {total_postprocess_time / num_images * 1000:.2f}ms")
    print(f"")
    print(f"⏱️  OVERALL PERFORMANCE:")
    print(f"   Total time: {total_processing_time:.2f}s")
    print(f"   Average per image: {total_processing_time / num_images * 1000:.2f}ms")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # CPU-side post-process: raw tensors from NPZ (fpga_inference.py on FPGA); output text matches ONNX flow.
    parser = argparse.ArgumentParser(
        description="NPZ post-processing for YOLOv8 detect (compiled / DPU int8 outputs)"
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to predictions.npz (from fpga_inference.py on the FPGA)",
    )
    parser.add_argument(
        "--pkl",
        required=True,
        help="Path to Vitis config .pkl (tensor_no, stride, reg_max, nc, ... from quantization)",
    )
    parser.add_argument("--iou", type=float, required=True, help="NMS IoU threshold")
    parser.add_argument(
        "--model-name",
        default="yolov8n",
        help="Label for logging (e.g., yolov8n)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output text path (default: output_boxes_<npz_stem>.txt in cwd)",
    )
    args = parser.parse_args()

    npz_stem = os.path.splitext(os.path.basename(args.npz))[0]
    out_path = args.output or f"output_boxes_{npz_stem}.txt"

    if not os.path.isfile(args.npz):
        print(f"ERROR: NPZ not found: {args.npz}")
        sys.exit(1)
    if not os.path.isfile(args.pkl):
        print(f"ERROR: Config pickle not found: {args.pkl}")
        sys.exit(1)

    postprocess_predictions(
        args.npz,
        args.pkl,
        args.model_name,
        args.iou,
        out_path,
    )
