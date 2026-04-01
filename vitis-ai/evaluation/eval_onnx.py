from __future__ import annotations

import argparse
import pickle
import cv2
import numpy as np
import onnxruntime as ort
import torch


from pathlib import Path

from utils import DetMetrics, OBBMetrics, TorchNMS, YAML, batch_probiou, box_iou, check_yaml


class DetectONNX:
    """ONNXRuntime evaluator for custom YOLOv26 Detect exports.

    Expected export layout from your DPU Detect head:
    - 3 outputs, one per level: [B, 4 * reg_max + nc, H, W]

    Detect forward inference branch:
    - DFL decode if reg_max > 1, otherwise direct ltrb distances
    - dist2bbox with xyxy (x1, y1, x2, y2)
    - sigmoid on class logits
    - class-aware NMS
    """

    def __init__(
        self,
        model: str,
        data: str | None,
        imgsz: int,
        conf: float,
        iou: float,
        nc: int,
        reg_max: int,
        strides: list[int],
        max_det: int,
    ) -> None:
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.max_det = max_det

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model, providers=providers or available)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        in_shape = input_meta.shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int) and in_shape[2] == in_shape[3]:
            fixed = int(in_shape[2])
            if self.imgsz != fixed:
                print(f"[INFO] Overriding imgsz from {self.imgsz} to ONNX input size {fixed}")
                self.imgsz = fixed

        self.names = self._load_names(data, nc)
        self.palette = np.random.default_rng(0).integers(0, 255, (len(self.names), 3), dtype=np.uint8)

    @staticmethod
    def _load_names(data: str | None, nc: int) -> list[str]:
        if not data:
            return [str(i) for i in range(nc)]
        yaml_file = check_yaml(data)
        if isinstance(yaml_file, (list, tuple)):
            yaml_file = yaml_file[0]
        names = YAML.load(str(yaml_file)).get("names", None)
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        if isinstance(names, list):
            return names
        return [str(i) for i in range(nc)]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def preprocess_image(self, im: np.ndarray) -> tuple[np.ndarray, dict]:
        h0, w0 = im.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

        resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.imgsz - new_w) / 2.0
        pad_h = (self.imgsz - new_h) / 2.0
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))

        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = out.astype(np.float32) / 255.0
        out = np.transpose(out, (2, 0, 1))[None]
        return out, {
            "mode": "letterbox",
            "h0": h0,
            "w0": w0,
            "gain": r,
            "pad_w": left,
            "pad_h": top,
        }

    @staticmethod
    def _to_bchw(x: np.ndarray, min_channels: int) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
        if x.shape[1] >= min_channels:
            return x
        if x.shape[-1] >= min_channels:
            return np.transpose(x, (0, 3, 1, 2))
        raise ValueError(f"Unable to infer channel dimension for shape {x.shape}")

    @staticmethod
    def _make_anchors(h: int, w: int) -> np.ndarray:
        sy, sx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5, np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
        return np.stack((sx, sy), axis=-1).reshape(-1, 2)

    def _split_outputs(self, outputs: list[np.ndarray]) -> list[np.ndarray]:
        if len(outputs) != 3:
            raise ValueError(f"Detect expects exactly 3 outputs, got {len(outputs)}")
        return outputs

    def decode(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        levels = self._split_outputs(outputs)
        dist_all, cls_all = [], []
        anchors_all, stride_all = [], []

        for i, out in enumerate(levels):
            min_c = 4 * self.reg_max + self.nc
            out = self._to_bchw(out, min_c)
            b, c, h, w = out.shape
            if b != 1:
                raise ValueError("This script currently supports batch size 1.")
            if c < min_c:
                raise ValueError(f"Level {i} has {c} channels, expected at least {min_c}.")

            dist_logits = out[:, : 4 * self.reg_max, :, :]
            cls_logits = out[:, 4 * self.reg_max : 4 * self.reg_max + self.nc, :, :]

            dist = dist_logits.reshape(1, 4, self.reg_max, h * w)
            cls = cls_logits.reshape(1, self.nc, h * w)

            dist_all.append(dist)
            cls_all.append(cls)
            anchors_all.append(self._make_anchors(h, w))
            stride_all.append(np.full((h * w, 1), float(self.strides[i]), dtype=np.float32))

        dist_logits = np.concatenate(dist_all, axis=3)
        cls_logits = np.concatenate(cls_all, axis=2)
        anchors = np.concatenate(anchors_all, axis=0)
        strides = np.concatenate(stride_all, axis=0)

        if self.reg_max > 1:
            proj = np.arange(self.reg_max, dtype=np.float32).reshape(1, 1, self.reg_max, 1)
            dist_prob = self._softmax(dist_logits, axis=2)
            pred_dist = np.sum(dist_prob * proj, axis=2)
        else:
            pred_dist = dist_logits.reshape(1, 4, -1)

        lt = np.transpose(pred_dist[:, 0:2, :], (0, 2, 1))
        rb = np.transpose(pred_dist[:, 2:4, :], (0, 2, 1))
        x1y1 = anchors[None] - lt
        x2y2 = anchors[None] + rb
        xyxy = np.concatenate((x1y1, x2y2), axis=-1)
        scale4 = np.repeat(strides, 4, axis=1)
        xyxy = xyxy * scale4[None]

        scores = self._sigmoid(np.transpose(cls_logits, (0, 2, 1)))
        return xyxy[0], scores[0]

    def postprocess(self, xyxy: np.ndarray, scores: np.ndarray, meta: dict) -> np.ndarray:
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1).astype(np.float32)
        keep = conf >= self.conf
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes = xyxy[keep]
        conf = conf[keep]
        cls = cls[keep]

        boxes_t = torch.from_numpy(boxes).float()
        scores_t = torch.from_numpy(conf).float()
        cls_t = torch.from_numpy(cls).float().unsqueeze(1)

        max_wh = 7680.0
        boxes_for_nms = boxes_t + cls_t * max_wh
        keep_idx = TorchNMS.nms(boxes_for_nms, scores_t, self.iou)
        keep_idx = keep_idx[: self.max_det].cpu().numpy()

        det = np.concatenate((boxes[keep_idx], conf[keep_idx, None], cls[keep_idx, None]), axis=1)

        if meta["mode"] == "resize":
            sx = meta["w0"] / self.imgsz
            sy = meta["h0"] / self.imgsz
            det[:, [0, 2]] *= sx
            det[:, [1, 3]] *= sy
        else:
            gain, pad_w, pad_h = meta["gain"], meta["pad_w"], meta["pad_h"]
            det[:, [0, 2]] = (det[:, [0, 2]] - pad_w) / gain
            det[:, [1, 3]] = (det[:, [1, 3]] - pad_h) / gain

        det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, meta["w0"])
        det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, meta["h0"])
        return det.astype(np.float32)

    def draw(self, image: np.ndarray, det: np.ndarray) -> np.ndarray:
        out = image.copy()
        for x1, y1, x2, y2, conf, cls in det:
            cls_i = int(cls)
            color = tuple(int(c) for c in self.palette[cls_i % len(self.palette)])
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.names[cls_i] if cls_i < len(self.names) else cls_i} {conf:.2f}"
            cv2.putText(out, label, (int(x1), max(int(y1) - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return out


class DetectMapEvaluator:
    """Detection mAP evaluator for ONNX predictions."""

    def __init__(self, names: list[str]) -> None:
        self.names = {i: n for i, n in enumerate(names)}
        self.metrics = DetMetrics(self.names)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

    def _match_predictions(self, pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        correct = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        if pred_cls.numel() == 0 or true_cls.numel() == 0:
            return torch.tensor(correct, dtype=torch.bool)

        correct_class = true_cls[:, None] == pred_cls
        iou = (iou * correct_class).cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool)

    @staticmethod
    def _resolve_label_path(image_path: Path) -> Path:
        p = image_path
        if "images" in p.parts:
            parts = list(p.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return p.with_suffix(".txt")

    @staticmethod
    def _xywhn_to_xyxy(vals: np.ndarray, h: int, w: int) -> np.ndarray:
        x, y, bw, bh = vals
        x *= w
        y *= h
        bw *= w
        bh *= h
        return np.array([x - bw / 2.0, y - bh / 2.0, x + bw / 2.0, y + bh / 2.0], dtype=np.float32)

    @staticmethod
    def _poly_to_xyxy(vals: np.ndarray, h: int, w: int) -> np.ndarray:
        xs = vals[0::2] * w
        ys = vals[1::2] * h
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    @staticmethod
    def _load_gt_boxes(image_path: Path, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load GT labels supporting detect (cls xywhn) and fallback OBB polygon (cls x1..y4)."""
        label_path = DetectMapEvaluator._resolve_label_path(image_path)
        if not label_path.exists():
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

        cls_list, box_list = [], []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                vals = np.array([float(x) for x in s.split()], dtype=np.float32)
                if vals.size >= 5 and vals.size < 9:
                    cls_list.append(vals[0])
                    box_list.append(DetectMapEvaluator._xywhn_to_xyxy(vals[1:5], h, w))
                elif vals.size >= 9:
                    cls_list.append(vals[0])
                    box_list.append(DetectMapEvaluator._poly_to_xyxy(vals[1:9], h, w))

        if not box_list:
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 4), dtype=torch.float32)

        return torch.tensor(cls_list, dtype=torch.float32), torch.tensor(np.stack(box_list), dtype=torch.float32)

    def update(self, image_path: Path, det: np.ndarray, im_shape: tuple[int, int]) -> None:
        h, w = im_shape
        tcls, tboxes = self._load_gt_boxes(image_path, h, w)

        if det.size == 0:
            tp = np.zeros((0, self.niou), dtype=bool)
            conf = np.zeros(0, dtype=np.float32)
            pred_cls = np.zeros(0, dtype=np.float32)
        else:
            p = torch.from_numpy(det).float()
            pboxes = p[:, :4]
            conf_t = p[:, 4]
            pcls_t = p[:, 5]
            if tboxes.shape[0] and pboxes.shape[0]:
                iou = box_iou(tboxes, pboxes)
                tp_t = self._match_predictions(pcls_t, tcls, iou)
                tp = tp_t.cpu().numpy()
            else:
                tp = np.zeros((pboxes.shape[0], self.niou), dtype=bool)
            conf = conf_t.cpu().numpy()
            pred_cls = pcls_t.cpu().numpy()

        self.metrics.update_stats(
            {
                "tp": tp,
                "conf": conf,
                "pred_cls": pred_cls,
                "target_cls": tcls.cpu().numpy(),
                "target_img": np.unique(tcls.cpu().numpy()),
            }
        )

    def finalize(self, save_dir: Path) -> dict[str, float]:
        self.metrics.process(save_dir=save_dir, plot=False)
        return self.metrics.results_dict


class OBB26ONNX:
    """ONNXRuntime evaluator for custom YOLOv26 OBB26 exports.

    Supports custom export layouts:
    - 3 outputs: one tensor per level, each containing [box_dfl, cls, angle_raw]
    - 6 outputs: first 3 tensors are [box_dfl, cls], last 3 tensors are [angle_raw]

    This decoder mirrors OBB26 math:
    - DFL decode for distances
    - angle = (sigmoid(raw) - 0.25) * pi
    - dist2rbox decode with per-level anchors/strides
    - class-wise rotated NMS
    """

    def __init__(
        self,
        model: str,
        data: str | None,
        imgsz: int,
        conf: float,
        iou: float,
        nc: int,
        reg_max: int,
        ne: int,
        strides: list[int],
        max_det: int,
    ) -> None:
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.reg_max = reg_max
        self.ne = ne
        self.strides = strides
        self.max_det = max_det
        self._resolved_angle_mode = None

        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(model, providers=providers or available)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        in_shape = input_meta.shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int) and in_shape[2] == in_shape[3]:
            fixed = int(in_shape[2])
            if self.imgsz != fixed:
                print(f"[INFO] Overriding imgsz from {self.imgsz} to ONNX input size {fixed}")
                self.imgsz = fixed

        self.names = self._load_names(data, nc)
        self.palette = np.random.default_rng(0).integers(0, 255, (len(self.names), 3), dtype=np.uint8)

    @staticmethod
    def _load_names(data: str | None, nc: int) -> list[str]:
        if not data:
            return [str(i) for i in range(nc)]
        yaml_file = check_yaml(data)
        if isinstance(yaml_file, (list, tuple)):
            yaml_file = yaml_file[0]
        names = YAML.load(str(yaml_file)).get("names", None)
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        if isinstance(names, list):
            return names
        return [str(i) for i in range(nc)]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    def preprocess_image(self, im: np.ndarray) -> tuple[np.ndarray, dict]:
        h0, w0 = im.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

        resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (self.imgsz - new_w) / 2.0
        pad_h = (self.imgsz - new_h) / 2.0
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))

        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = out.astype(np.float32) / 255.0
        out = np.transpose(out, (2, 0, 1))[None]

        meta = {
            "mode": "letterbox",
            "h0": h0,
            "w0": w0,
            "gain": r,
            "pad_w": left,
            "pad_h": top,
        }
        return out, meta

    @staticmethod
    def _to_bchw(x: np.ndarray, min_channels: int) -> np.ndarray:
        """Convert BCHW/BHWC tensor to BCHW using channel heuristics."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
        if x.shape[1] >= min_channels:
            return x
        if x.shape[-1] >= min_channels:
            return np.transpose(x, (0, 3, 1, 2))
        raise ValueError(f"Unable to infer channel dimension for shape {x.shape}")

    @staticmethod
    def _make_anchors(h: int, w: int) -> np.ndarray:
        sy, sx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5, np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
        return np.stack((sx, sy), axis=-1).reshape(-1, 2)

    def _split_outputs(self, outputs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if len(outputs) != 6:
            raise ValueError(f"OBB expects exactly 6 outputs (split6), got {len(outputs)}")
        boxcls = outputs[:3]
        angles = outputs[3:]
        return boxcls, angles

    def decode(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxcls_lvls, angle_lvls = self._split_outputs(outputs)

        dist_all, cls_all, angle_all = [], [], []
        anchors_all, stride_all = [], []

        for i, boxcls in enumerate(boxcls_lvls):
            min_c = 4 * self.reg_max + self.nc
            boxcls = self._to_bchw(boxcls, min_c)
            b, c, h, w = boxcls.shape
            if b != 1:
                raise ValueError("This script currently supports batch size 1.")
            box_ch = 4 * self.reg_max
            cls_ch = self.nc
            angle = self._to_bchw(angle_lvls[i], self.ne)
            if angle.shape[1] != self.ne:
                raise ValueError(f"Angle channels mismatch: expected {self.ne}, got {angle.shape[1]}")

            dist_logits = boxcls[:, :box_ch, :, :]
            cls_logits = boxcls[:, box_ch : box_ch + cls_ch, :, :]
            angle_logits = angle

            dist = dist_logits.reshape(1, 4, self.reg_max, h * w)
            cls = cls_logits.reshape(1, self.nc, h * w)
            ang = angle_logits.reshape(1, self.ne, h * w)

            dist_all.append(dist)
            cls_all.append(cls)
            angle_all.append(ang)
            anchors_all.append(self._make_anchors(h, w))
            stride_all.append(np.full((h * w, 1), float(self.strides[i]), dtype=np.float32))

        dist_logits = np.concatenate(dist_all, axis=3)
        cls_logits = np.concatenate(cls_all, axis=2)
        angle_logits = np.concatenate(angle_all, axis=2)
        anchors = np.concatenate(anchors_all, axis=0)
        strides = np.concatenate(stride_all, axis=0)

        if self.reg_max > 1:
            proj = np.arange(self.reg_max, dtype=np.float32).reshape(1, 1, self.reg_max, 1)
            dist_prob = self._softmax(dist_logits, axis=2)
            pred_dist = np.sum(dist_prob * proj, axis=2)
        else:
            pred_dist = dist_logits.reshape(1, 4, -1)

        amin = float(angle_logits.min())
        amax = float(angle_logits.max())
        lo, hi = -0.25 * np.pi, 0.75 * np.pi
        mode = "normalized" if amin >= (lo - 0.25) and amax <= (hi + 0.25) else "raw"

        if self._resolved_angle_mode is None:
            self._resolved_angle_mode = mode
            print(f"[INFO] Angle decode mode: {mode}")

        if mode == "normalized":
            pred_angle = angle_logits
        else:
            pred_angle = (self._sigmoid(angle_logits) - 0.25) * np.pi

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

        xywh = np.concatenate((xy, wh), axis=-1) * strides[None]
        cls_scores = self._sigmoid(np.transpose(cls_logits, (0, 2, 1)))
        angles = ang

        return xywh[0], cls_scores[0], angles[0]

    def postprocess(self, xywh: np.ndarray, scores: np.ndarray, angles: np.ndarray, meta: dict) -> np.ndarray:
        conf = scores.max(axis=1)
        cls = scores.argmax(axis=1).astype(np.float32)
        keep = conf >= self.conf

        if not np.any(keep):
            return np.zeros((0, 7), dtype=np.float32)

        xywh = xywh[keep]
        conf = conf[keep]
        cls = cls[keep]
        angles = angles[keep]

        boxes_t = torch.from_numpy(np.concatenate((xywh, angles), axis=1)).float()
        scores_t = torch.from_numpy(conf).float()
        cls_t = torch.from_numpy(cls).float().unsqueeze(1)

        max_wh = 7680.0
        offset = cls_t * max_wh
        nms_boxes = torch.cat((boxes_t[:, :2] + offset, boxes_t[:, 2:4], boxes_t[:, 4:5]), dim=1)
        keep_idx = TorchNMS.fast_nms(nms_boxes, scores_t, self.iou, iou_func=batch_probiou)
        keep_idx = keep_idx[: self.max_det].cpu().numpy()

        det = np.concatenate(
            (
                xywh[keep_idx],
                conf[keep_idx, None],
                cls[keep_idx, None],
                angles[keep_idx],
            ),
            axis=1,
        )

        if meta["mode"] == "resize":
            sx = meta["w0"] / self.imgsz
            sy = meta["h0"] / self.imgsz
            det[:, 0] *= sx
            det[:, 1] *= sy
            det[:, 2] *= sx
            det[:, 3] *= sy
        else:
            gain, pad_w, pad_h = meta["gain"], meta["pad_w"], meta["pad_h"]
            det[:, 0] = (det[:, 0] - pad_w) / gain
            det[:, 1] = (det[:, 1] - pad_h) / gain
            det[:, 2] /= gain
            det[:, 3] /= gain

        det[:, 0] = np.clip(det[:, 0], 0, meta["w0"])
        det[:, 1] = np.clip(det[:, 1], 0, meta["h0"])
        det[:, 2] = np.clip(det[:, 2], 0, meta["w0"])
        det[:, 3] = np.clip(det[:, 3], 0, meta["h0"])

        return det.astype(np.float32)

    def draw(self, image: np.ndarray, det: np.ndarray) -> np.ndarray:
        out = image.copy()
        for x, y, w, h, conf, cls, angle in det:
            cls_i = int(cls)
            color = tuple(int(c) for c in self.palette[cls_i % len(self.palette)])

            rect = ((float(x), float(y)), (max(float(w), 1.0), max(float(h), 1.0)), float(angle * 180.0 / np.pi))
            pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

            label = f"{self.names[cls_i] if cls_i < len(self.names) else cls_i} {conf:.2f}"
            lx, ly = int(pts[0][0]), int(pts[0][1]) - 6
            cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return out

    def run_one(self, image_path: Path, save_dir: Path, save_txt: bool = False) -> None:
        im0 = cv2.imread(str(image_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {image_path}")
            return

        x, meta = self.preprocess_image(im0)
        outputs = self.session.run(None, {self.input_name: x})
        outputs = [o.astype(np.float32) for o in outputs]

        xywh, cls_scores, angles = self.decode(outputs)
        det = self.postprocess(xywh, cls_scores, angles, meta)

        vis = self.draw(im0, det)
        out_file = save_dir / image_path.name
        cv2.imwrite(str(out_file), vis)

        if save_txt:
            txt_file = save_dir / f"{image_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{image_path.name}: {len(det)} detections -> {out_file}")


def xyxyxyxy2xywhr(x):
    """Convert OBB corners [x1,y1,...,x4,y4] to [cx,cy,w,h,theta]."""
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        theta = angle / 180 * np.pi
        if w < h:
            w, h = h, w
            theta += np.pi / 2
        while theta >= 3 * np.pi / 4:
            theta -= np.pi
        while theta < -np.pi / 4:
            theta += np.pi
        rboxes.append([cx, cy, w, h, theta])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


class OBBMapEvaluator:
    """OBB mAP evaluator for ONNX predictions."""

    def __init__(self, names: list[str]) -> None:
        self.names = {i: n for i, n in enumerate(names)}
        self.metrics = OBBMetrics(self.names)
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()

    def _match_predictions(self, pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        """Match predictions to GT using greedy IoU assignment per threshold."""
        correct = np.zeros((pred_cls.shape[0], self.niou), dtype=bool)
        if pred_cls.numel() == 0 or true_cls.numel() == 0:
            return torch.tensor(correct, dtype=torch.bool)

        correct_class = true_cls[:, None] == pred_cls
        iou = (iou * correct_class).cpu().numpy()

        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool)

    @staticmethod
    def _resolve_label_path(image_path: Path) -> Path:
        p = image_path
        if "images" in p.parts:
            parts = list(p.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return p.with_suffix(".txt")

    @staticmethod
    def _load_gt_obb(image_path: Path, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load OBB labels in YOLO OBB txt format: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
        label_path = OBBMapEvaluator._resolve_label_path(image_path)
        if not label_path.exists():
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 5), dtype=torch.float32)

        rows = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                vals = [float(x) for x in s.split()]
                if len(vals) >= 9:
                    rows.append(vals[:9])

        if not rows:
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0, 5), dtype=torch.float32)

        arr = np.array(rows, dtype=np.float32)
        cls = torch.from_numpy(arr[:, 0])
        polys = torch.from_numpy(arr[:, 1:9])
        polys[:, 0::2] *= float(w)
        polys[:, 1::2] *= float(h)
        obb = xyxyxyxy2xywhr(polys)
        return cls, obb

    def update(self, image_path: Path, det: np.ndarray, im_shape: tuple[int, int]) -> None:
        """Update mAP stats for one image.

        det format: Nx7 [x, y, w, h, conf, cls, angle]
        """
        h, w = im_shape
        tcls, tbboxes = self._load_gt_obb(image_path, h, w)

        if det.size == 0:
            tp = np.zeros((0, self.niou), dtype=bool)
            conf = np.zeros(0, dtype=np.float32)
            pred_cls = np.zeros(0, dtype=np.float32)
        else:
            p = torch.from_numpy(det).float()
            pboxes = p[:, [0, 1, 2, 3, 6]]
            conf_t = p[:, 4]
            pcls_t = p[:, 5]
            if tbboxes.shape[0] and pboxes.shape[0]:
                iou = batch_probiou(tbboxes, pboxes)
                tp_t = self._match_predictions(pcls_t, tcls, iou)
                tp = tp_t.cpu().numpy()
            else:
                tp = np.zeros((pboxes.shape[0], self.niou), dtype=bool)
            conf = conf_t.cpu().numpy()
            pred_cls = pcls_t.cpu().numpy()

        self.metrics.update_stats(
            {
                "tp": tp,
                "conf": conf,
                "pred_cls": pred_cls,
                "target_cls": tcls.cpu().numpy(),
                "target_img": np.unique(tcls.cpu().numpy()),
            }
        )

    def finalize(self, save_dir: Path) -> dict[str, float]:
        self.metrics.process(save_dir=save_dir, plot=False)
        return self.metrics.results_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate custom YOLOv26 ONNX outputs for OBB or Detect.")
    parser.add_argument(
        "--task",
        type=str,
        default="obb",
        choices=("obb", "detect"),
        help="Task type: 'obb' for oriented boxes or 'detect' for axis-aligned detection.",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML for class names and validation images")
    parser.add_argument(
        "--quant-meta",
        type=str,
        default=None,
        help="Path to quantization metadata pkl from run_model_exports (no, stride, reg_max, nc, dfl)",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="Rotated NMS IoU threshold")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes")
    parser.add_argument("--reg-max", type=int, default=16, help="DFL reg_max")
    parser.add_argument("--ne", type=int, default=1, help="Angle channels")
    parser.add_argument("--strides", type=str, default="8,16,32", help="Comma-separated strides per output level")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save detections as txt. detect: x1 y1 x2 y2 conf cls, obb: x y w h conf cls angle",
    )
    parser.add_argument("--eval-map", action="store_true", help="Compute mAP50 and mAP50-95")
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def resolve_source_from_data(data_arg: str | None) -> Path:
    """Resolve validation image path from dataset yaml.

    Priority: val -> test. If a list is provided, uses the first entry.
    """
    if not data_arg:
        raise ValueError("--data is required.")

    yaml_file = check_yaml(data_arg)
    if isinstance(yaml_file, (list, tuple)):
        yaml_file = yaml_file[0]
    yaml_path = Path(str(yaml_file))
    cfg = YAML.load(str(yaml_path))

    src = cfg.get("val", None)
    if src is None:
        src = cfg.get("test", None)
    if src is None:
        raise ValueError("Could not find 'val' or 'test' entries in data yaml.")

    if isinstance(src, (list, tuple)):
        if not src:
            raise ValueError("Data yaml contains empty 'val'/'test' list.")
        src = src[0]

    src_path = Path(str(src))
    if not src_path.is_absolute():
        src_path = (yaml_path.parent / src_path).resolve()

    return src_path


def apply_quant_meta(args: argparse.Namespace) -> None:
    """Override model decode settings from quantization metadata pkl when provided."""
    if not args.quant_meta:
        return

    with open(args.quant_meta, "rb") as f:
        no, stride, reg_max, nc, _dfl = pickle.load(f)

    args.reg_max = int(reg_max)
    args.nc = int(nc)

    if isinstance(stride, torch.Tensor):
        stride_vals = stride.detach().cpu().tolist()
    elif isinstance(stride, (list, tuple)):
        stride_vals = list(stride)
    else:
        stride_vals = [stride]

    args.strides = ",".join(str(int(s)) for s in stride_vals)
    print(
        f"Loaded quant meta from {args.quant_meta}: no={int(no)}, reg_max={args.reg_max}, nc={args.nc}, "
        f"strides={args.strides}"
    )


def run_obb(args: argparse.Namespace, images: list[Path], save_dir: Path, source: Path, strides: list[int]) -> None:
    evaluator = OBB26ONNX(
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nc=args.nc,
        reg_max=args.reg_max,
        ne=args.ne,
        strides=strides,
        max_det=args.max_det,
    )

    print(f"Loaded model: {args.model}")
    print(f"Task: OBB")
    print(f"Input source: {source}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {save_dir}")

    map_eval = OBBMapEvaluator(evaluator.names) if args.eval_map else None

    for im_path in images:
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        x, meta = evaluator.preprocess_image(im0)
        outputs = evaluator.session.run(None, {evaluator.input_name: x})
        outputs = [o.astype(np.float32) for o in outputs]
        xywh, cls_scores, angles = evaluator.decode(outputs)
        det = evaluator.postprocess(xywh, cls_scores, angles, meta)

        vis = evaluator.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")

        if map_eval is not None:
            map_eval.update(im_path, det, im0.shape[:2])

    if map_eval is not None:
        results = map_eval.finalize(save_dir)
        print("\n===== OBB mAP Results =====")
        print(f"Precision:  {results.get('metrics/precision(B)', 0.0):.6f}")
        print(f"Recall:     {results.get('metrics/recall(B)', 0.0):.6f}")
        print(f"mAP50:      {results.get('metrics/mAP50(B)', 0.0):.6f}")
        print(f"mAP50-95:   {results.get('metrics/mAP50-95(B)', 0.0):.6f}")


def run_detect(args: argparse.Namespace, images: list[Path], save_dir: Path, source: Path, strides: list[int]) -> None:
    evaluator = DetectONNX(
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        nc=args.nc,
        reg_max=args.reg_max,
        strides=strides,
        max_det=args.max_det,
    )

    print(f"Loaded model: {args.model}")
    print(f"Task: Detect")
    print(f"Input source: {source}")
    print(f"Images found: {len(images)}")
    print(f"Output dir: {save_dir}")

    map_eval = DetectMapEvaluator(evaluator.names) if args.eval_map else None

    for im_path in images:
        im0 = cv2.imread(str(im_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {im_path}")
            continue

        x, meta = evaluator.preprocess_image(im0)
        outputs = evaluator.session.run(None, {evaluator.input_name: x})
        outputs = [o.astype(np.float32) for o in outputs]
        xyxy, scores = evaluator.decode(outputs)
        det = evaluator.postprocess(xyxy, scores, meta)

        vis = evaluator.draw(im0, det)
        out_file = save_dir / im_path.name
        cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{im_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        print(f"{im_path.name}: {len(det)} detections -> {out_file}")

        if map_eval is not None:
            map_eval.update(im_path, det, im0.shape[:2])

    if map_eval is not None:
        results = map_eval.finalize(save_dir)
        print("\n===== Detect mAP Results =====")
        print(f"Precision:  {results.get('metrics/precision(B)', 0.0):.6f}")
        print(f"Recall:     {results.get('metrics/recall(B)', 0.0):.6f}")
        print(f"mAP50:      {results.get('metrics/mAP50(B)', 0.0):.6f}")
        print(f"mAP50-95:   {results.get('metrics/mAP50-95(B)', 0.0):.6f}")


def main() -> None:
    args = parse_args()
    apply_quant_meta(args)

    strides = [int(s.strip()) for s in args.strides.split(",") if s.strip()]
    if len(strides) != 3:
        raise ValueError(f"Expected 3 strides, got {strides}")

    source = resolve_source_from_data(args.data)
    images = iter_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in source: {source}")

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path("runs/detect_onnx_eval") if args.task == "detect" else Path("runs/obb26_onnx_eval")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "detect":
        run_detect(args, images, save_dir, source, strides)
    else:
        run_obb(args, images, save_dir, source, strides)


if __name__ == "__main__":
    main()
