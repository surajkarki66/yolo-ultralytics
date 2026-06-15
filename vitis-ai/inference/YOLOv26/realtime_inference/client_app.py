import argparse
import pickle
import socket
import struct
import time

import cv2
import numpy as np

FRAME_ID_HEADER = struct.Struct("!I")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def make_anchors(h, w):
    sy, sx = np.meshgrid(
        np.arange(h, dtype=np.float32) + 0.5,
        np.arange(w, dtype=np.float32) + 0.5,
        indexing="ij",
    )
    return np.stack((sx, sy), axis=-1).reshape(-1, 2)


def bbox_iou_xyxy(box, boxes):
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-7
    return inter / union


def nms_class_aware(xyxy, scores, conf_thres, iou_thres, max_det=200):
    conf = scores.max(axis=1)
    cls = scores.argmax(axis=1).astype(np.float32)
    keep = conf >= conf_thres
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)
    boxes = xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]
    order = np.argsort(-conf)
    det = np.concatenate([boxes, conf[:, None], cls[:, None]], axis=1)
    out = []
    while order.size > 0 and len(out) < max_det:
        i = order[0]
        out.append(det[i])
        if order.size == 1:
            break
        rest = order[1:]
        same_cls = cls[rest] == cls[i]
        ious = bbox_iou_xyxy(boxes[i], boxes[rest])
        rest = rest[np.logical_or(~same_cls, ious <= iou_thres)]
        order = rest
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 6), dtype=np.float32)


def _build_level_order(out_shapes, expected_strides, img_size):
    hw = [(o[1], o[2]) for o in out_shapes]
    stride_est = [float(img_size) / h for h, _ in hw]
    rem = set(range(len(out_shapes)))
    reorder = []
    level_hws = []
    for s in expected_strides:
        idx = min(rem, key=lambda x: abs(stride_est[x] - float(s)))
        rem.remove(idx)
        reorder.append(idx)
        level_hws.append(hw[idx])
    return reorder, level_hws


def build_detect_ctx(out_shapes, expected_strides, img_size):
    reorder, level_hws = _build_level_order(out_shapes, expected_strides, img_size)
    anchors_parts, scale_parts, starts = [], [], []
    total = 0
    for i, (h, w) in enumerate(level_hws):
        starts.append(total)
        a = h * w
        anchors_parts.append(make_anchors(h, w))
        scale_parts.append(np.full((a,), float(expected_strides[i]), dtype=np.float32))
        total += a
    return {
        "reorder": reorder,
        "level_hws": level_hws,
        "starts": starts,
        "anchors": np.concatenate(anchors_parts, axis=0),
        "scale4": np.repeat(np.concatenate(scale_parts)[:, None], 4, axis=1),
        "total": total,
    }


def decode_detect_outputs(out_data, out_scales, reg_max, nc, ctx):
    deq = [arr.astype(np.float32) / out_scales[i] for i, arr in enumerate(out_data)]
    reorder = ctx["reorder"]
    level_hws = ctx["level_hws"]
    starts = ctx["starts"]
    anchors = ctx["anchors"]
    scale4 = ctx["scale4"]
    total = ctx["total"]

    dist_logits = np.empty((1, 4, reg_max, total), dtype=np.float32)
    cls_logits = np.empty((1, nc, total), dtype=np.float32)

    for li, oi in enumerate(reorder):
        h, w = level_hws[li]
        a = h * w
        start = starts[li]
        out = deq[oi].transpose(0, 3, 1, 2)  # NHWC -> NCHW
        dist = out[:, :4 * reg_max, :, :].reshape(1, 4, reg_max, a)
        cls = out[:, 4 * reg_max: 4 * reg_max + nc, :, :].reshape(1, nc, a)
        dist_logits[:, :, :, start:start + a] = dist
        cls_logits[:, :, start:start + a] = cls

    if reg_max > 1:
        proj = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        pred_dist = np.sum(softmax(dist_logits, axis=2) * proj, axis=2)
    else:
        pred_dist = dist_logits.reshape(1, 4, -1)

    lt = np.transpose(pred_dist[:, 0:2, :], (0, 2, 1))
    rb = np.transpose(pred_dist[:, 2:4, :], (0, 2, 1))
    x1y1 = anchors[None] - lt
    x2y2 = anchors[None] + rb
    xyxy = np.concatenate((x1y1, x2y2), axis=-1) * scale4[None]
    scores = sigmoid(np.transpose(cls_logits, (0, 2, 1)))
    return xyxy[0].astype(np.float32), scores[0].astype(np.float32)


def _get_covariance_matrix(boxes):
    gbbs = np.concatenate([(boxes[..., 2:4] ** 2) / 12.0, boxes[..., 4:5]], axis=-1)
    a, b, c = np.split(gbbs, 3, axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )


def batch_probiou(obb1, obb2, eps=1e-7):
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


def rotated_nms_class_aware(xywha, scores, conf_threshold, iou_threshold, max_det=200):
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


def build_obb_ctx(out_shapes, expected_strides, img_size):
    if len(out_shapes) != 6:
        raise ValueError(f"OBB expects 6 outputs, got {len(out_shapes)}")
    box_idxs = []
    angle_idxs = []
    for i, shape in enumerate(out_shapes):
        c = int(shape[3])  # NHWC
        if c > 1:
            box_idxs.append(i)
        else:
            angle_idxs.append(i)
    if len(box_idxs) != 3 or len(angle_idxs) != 3:
        raise ValueError(f"Could not split OBB outputs into 3 box + 3 angle heads: {out_shapes}")
    box_reorder, level_hws = _build_level_order([out_shapes[i] for i in box_idxs], expected_strides, img_size)
    angle_reorder, _ = _build_level_order([out_shapes[i] for i in angle_idxs], expected_strides, img_size)
    box_indices = [box_idxs[i] for i in box_reorder]
    angle_indices = [angle_idxs[i] for i in angle_reorder]
    starts, anchors_parts, stride_parts = [], [], []
    total = 0
    for i, (h, w) in enumerate(level_hws):
        starts.append(total)
        a = h * w
        anchors_parts.append(make_anchors(h, w))
        stride_parts.append(np.full((a,), float(expected_strides[i]), dtype=np.float32))
        total += a
    return {
        "box_indices": box_indices,
        "angle_indices": angle_indices,
        "level_hws": level_hws,
        "starts": starts,
        "anchors": np.concatenate(anchors_parts, axis=0),
        "stride_cat": np.concatenate(stride_parts, axis=0),
        "total": total,
    }


def decode_obb_outputs(out_data, out_scales, reg_max, nc, ctx):
    out_dequant = [arr.astype(np.float32) / out_scales[i] for i, arr in enumerate(out_data)]
    box_indices = ctx["box_indices"]
    angle_indices = ctx["angle_indices"]
    level_hws = ctx["level_hws"]
    starts = ctx["starts"]
    total = ctx["total"]
    anchors = ctx["anchors"]
    stride_cat = ctx["stride_cat"]
    dist_logits_cat = np.empty((1, 4, reg_max, total), dtype=np.float32)
    cls_logits_cat = np.empty((1, nc, total), dtype=np.float32)
    angle_logits_cat = np.empty((1, 1, total), dtype=np.float32)
    for li, (bidx, aidx) in enumerate(zip(box_indices, angle_indices)):
        h, w = level_hws[li]
        a = h * w
        start = starts[li]
        box_out = out_dequant[bidx].transpose(0, 3, 1, 2)
        ang_out = out_dequant[aidx].transpose(0, 3, 1, 2)
        dist_logits_cat[:, :, :, start:start + a] = box_out[:, : 4 * reg_max, :, :].reshape(1, 4, reg_max, a)
        cls_logits_cat[:, :, start:start + a] = box_out[:, 4 * reg_max : 4 * reg_max + nc, :, :].reshape(1, nc, a)
        angle_logits_cat[:, :, start:start + a] = ang_out[:, :1, :, :].reshape(1, 1, a)

    if reg_max > 1:
        proj = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        pred_dist = np.sum(softmax(dist_logits_cat, axis=2) * proj, axis=2)
    else:
        pred_dist = dist_logits_cat.reshape(1, 4, -1)

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
    xy = np.concatenate((x, y), axis=-1) + anchors[None]
    wh = lt + rb
    xywh = np.concatenate((xy, wh), axis=-1) * stride_cat[None, :, None]
    scores = sigmoid(np.transpose(cls_logits_cat, (0, 2, 1)))
    xywha = np.concatenate([xywh[0], ang[0]], axis=1).astype(np.float32)
    return xywha, scores[0].astype(np.float32)


class YOLOv26Client:
    def __init__(self, args):
        self.fpga_host = args.fpga_host
        self.fpga_port = args.fpga_port
        self.iou_threshold = args.iou_threshold
        self.conf_threshold = args.conf_threshold
        self.img_size = args.img_size
        self.camera_id = args.camera_id
        self.reg_max = args.reg_max
        self.nc = args.nc
        self.expected_strides = [int(x) for x in args.strides.split(",")]
        self.task = args.task
        self.input_scale = 1.0
        self.output_scales = None
        self.output_shapes = None
        self.input_shape = None
        self.decode_ctx = None
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.socket.connect((self.fpga_host, self.fpga_port))
            print(f"Connected to FPGA at {self.fpga_host}:{self.fpga_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to FPGA: {e}")
            return False

    def disconnect(self):
        self.socket.close()
        print("Disconnected from FPGA server")

    def recv_exact(self, n):
        data = b""
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return b""
            data += chunk
        return data

    def send_frame(self, frame):
        frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
        input_tensor = (frame_resized.astype(np.float32) / 255.0)
        input_tensor = (input_tensor * self.input_scale).astype(np.int8)
        frame_id = int(time.time() * 1000) & 0x7FFFFFFF
        request_bytes = FRAME_ID_HEADER.pack(frame_id) + np.ascontiguousarray(input_tensor, dtype=np.int8).tobytes()
        self.socket.sendall(struct.pack("!I", len(request_bytes)) + request_bytes)

        size_data = self.recv_exact(4)
        if not size_data:
            return None
        result_size = struct.unpack("!I", size_data)[0]
        result_data = self.recv_exact(result_size)
        if len(result_data) != result_size:
            return None
        if len(result_data) < FRAME_ID_HEADER.size:
            return None
        _resp_frame_id = FRAME_ID_HEADER.unpack(result_data[: FRAME_ID_HEADER.size])[0]
        raw = result_data[FRAME_ID_HEADER.size :]
        outputs = []
        offset = 0
        for shape in self.output_shapes:
            n = int(np.prod(shape))
            nbytes = n  # int8
            chunk = raw[offset : offset + nbytes]
            if len(chunk) != nbytes:
                return None
            arr = np.frombuffer(chunk, dtype=np.int8).reshape(shape)
            outputs.append(arr)
            offset += nbytes
        return outputs

    def post_process_tensors(self, output_tensors):
        if self.task == "obb":
            xywha, scores = decode_obb_outputs(
                out_data=output_tensors,
                out_scales=self.output_scales,
                reg_max=self.reg_max,
                nc=self.nc,
                ctx=self.decode_ctx,
            )
            det = rotated_nms_class_aware(
                xywha=xywha,
                scores=scores,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
            return det
        xyxy, scores = decode_detect_outputs(
            out_data=output_tensors,
            out_scales=self.output_scales,
            reg_max=self.reg_max,
            nc=self.nc,
            ctx=self.decode_ctx,
        )
        det = nms_class_aware(xyxy=xyxy, scores=scores, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)
        return [[x1, y1, x2 - x1, y2 - y1, conf] for x1, y1, x2, y2, conf, _ in det]

    def draw_detections(self, frame, detections):
        if detections is None:
            return frame
        if isinstance(detections, np.ndarray):
            if detections.size == 0:
                return frame
        elif len(detections) == 0:
            return frame
        h, w = frame.shape[:2]
        sx = w / self.img_size
        sy = h / self.img_size
        if self.task == "obb":
            for x, y, bw, bh, conf, _cls, angle in detections:
                x *= sx
                y *= sy
                bw *= sx
                bh *= sy
                rect = ((float(x), float(y)), (max(float(bw), 1.0), max(float(bh), 1.0)), float(angle * 180.0 / np.pi))
                pts = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (int(pts[0][0]), max(int(pts[0][1]) - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            return frame
        for x, y, bw, bh, conf in detections:
            x = int(x * sx)
            y = int(y * sy)
            bw = int(bw * sx)
            bh = int(bh * sy)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = f"person {conf:.2f}"
            cv2.putText(frame, label, (x, max(y - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def run(self):
        if not self.connect():
            return

        meta_size = self.recv_exact(4)
        if not meta_size:
            print("Failed to receive server metadata")
            return
        meta_len = struct.unpack("!I", meta_size)[0]
        meta_raw = self.recv_exact(meta_len)
        meta = pickle.loads(meta_raw)
        server_task = str(meta.get("task", "detect"))
        if self.task == "auto":
            self.task = server_task
        self.input_shape = tuple(meta.get("input_shape"))
        self.input_scale = float(meta.get("input_scale", 1.0))
        self.output_scales = [float(v) for v in meta.get("output_scales", [1.0, 1.0, 1.0])]
        self.output_shapes = [tuple(s) for s in meta.get("output_shapes", [])]
        print(f"Server meta: task={server_task}, using_task={self.task}, input_shape={self.input_shape}, output_shapes={self.output_shapes}")
        print(f"Server meta: input_scale={self.input_scale}, output_scales={self.output_scales}")
        if self.task == "obb":
            self.decode_ctx = build_obb_ctx(self.output_shapes, self.expected_strides, self.img_size)
        else:
            self.decode_ctx = build_detect_ctx(self.output_shapes, self.expected_strides, self.img_size)

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Cannot open camera (ID: {self.camera_id})")
            self.disconnect()
            return

        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                output_tensors = self.send_frame(frame)
                if output_tensors:
                    detections = self.post_process_tensors(output_tensors)
                    out_frame = self.draw_detections(frame.copy(), detections)
                    frame_count += 1
                    fps = frame_count / max(time.time() - start_time, 1e-6)
                    cv2.putText(out_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("YOLOv26 Real-time Detection", out_frame)
                else:
                    cv2.imshow("YOLOv26 Real-time Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv26 socket client (post-processing on laptop)")
    parser.add_argument("--fpga-host", type=str, required=True, help="IP address of FPGA server")
    parser.add_argument("--fpga-port", type=int, default=8888, help="FPGA server port")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device id")
    parser.add_argument("--img-size", type=int, default=416, help="Model input size")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.25, help="IoU threshold")
    parser.add_argument("--reg-max", type=int, default=1, help="YOLOv26 reg_max")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    parser.add_argument("--strides", type=str, default="8,16,32", help="Comma separated strides")
    parser.add_argument("--task", type=str, choices=["auto", "detect", "obb"], default="auto", help="Detection head type")
    return parser.parse_args()


if __name__ == "__main__":
    YOLOv26Client(parse_args()).run()
