#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from utils import DetMetrics, OBBMetrics, TorchNMS, YAML, batch_probiou, box_iou, check_yaml


def xyxyxyxy2xywhr(x):
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


class DetectMapEvaluator:
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

    def finalize(self) -> dict[str, float]:
        self.metrics.process()
        return self.metrics.results_dict


class OBBMapEvaluator:
    def __init__(self, names: list[str]) -> None:
        self.names = {i: n for i, n in enumerate(names)}
        self.metrics = OBBMetrics(self.names)
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
    def _load_gt_obb(image_path: Path, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def finalize(self) -> dict[str, float]:
        self.metrics.process()
        return self.metrics.results_dict


class DetectFromNPZ:
    def __init__(
        self,
        data: str | None,
        imgsz: int,
        conf: float,
        iou: float,
        nc: int,
        reg_max: int,
        strides: list[int],
        max_det: int,
    ) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.max_det = max_det

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

    @staticmethod
    def _to_bchw(x: np.ndarray, min_channels: int) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")

        if x.shape[1] == min_channels:
            return x

        if x.shape[-1] == min_channels:
            return np.transpose(x, (0, 3, 1, 2))

        c1 = int(x.shape[1])
        c_last = int(x.shape[-1])
        if c1 >= min_channels and c_last < min_channels:
            return x
        if c_last >= min_channels and c1 < min_channels:
            return np.transpose(x, (0, 3, 1, 2))

        if abs(c1 - min_channels) <= abs(c_last - min_channels):
            return x
        return np.transpose(x, (0, 3, 1, 2))

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

        min_c = 4 * self.reg_max + self.nc
        reordered = []
        for s in self.strides:
            best = None
            for idx, out in enumerate(levels):
                out_bchw = self._to_bchw(out, min_c)
                _, _c, h, _w = out_bchw.shape
                stride_est = float(self.imgsz) / float(h)
                diff = abs(stride_est - float(s))
                if best is None or diff < best[0]:
                    best = (diff, idx)
            assert best is not None
            reordered.append(levels[best[1]])
        levels = reordered

        dist_all, cls_all = [], []
        anchors_all, stride_all = [], []

        for i, out in enumerate(levels):
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

        sx = meta["w0"] / self.imgsz
        sy = meta["h0"] / self.imgsz
        det[:, [0, 2]] *= sx
        det[:, [1, 3]] *= sy

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
            cv2.putText(
                out,
                label,
                (int(x1), max(int(y1) - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return out


class OBB26FromNPZ:
    def __init__(
        self,
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
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nc = nc
        self.reg_max = reg_max
        self.ne = ne
        self.strides = strides
        self.max_det = max_det
        self._resolved_angle_mode = None

        self.names = DetectFromNPZ._load_names(data, nc)
        self.palette = np.random.default_rng(0).integers(0, 255, (len(self.names), 3), dtype=np.uint8)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        x = x - np.max(x, axis=axis, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=axis, keepdims=True)

    @staticmethod
    def _to_bchw(x: np.ndarray, min_channels: int) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
        if x.shape[1] == min_channels:
            return x
        if x.shape[-1] == min_channels:
            return np.transpose(x, (0, 3, 1, 2))

        c1 = int(x.shape[1])
        c_last = int(x.shape[-1])
        if c1 >= min_channels and c_last < min_channels:
            return x
        if c_last >= min_channels and c1 < min_channels:
            return np.transpose(x, (0, 3, 1, 2))

        if abs(c1 - min_channels) <= abs(c_last - min_channels):
            return x
        return np.transpose(x, (0, 3, 1, 2))

    @staticmethod
    def _make_anchors(h: int, w: int) -> np.ndarray:
        sy, sx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5, np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
        return np.stack((sx, sy), axis=-1).reshape(-1, 2)

    def _split_outputs(self, outputs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if len(outputs) != 6:
            raise ValueError(f"OBB expects exactly 6 outputs (split6), got {len(outputs)}")

        box_ch = 4 * self.reg_max + self.nc

        box_idxs: list[int] = []
        angle_idxs: list[int] = []

        for i, out in enumerate(outputs):
            if out.ndim != 4:
                continue
            c1 = int(out.shape[1])
            clast = int(out.shape[-1])
            if c1 == box_ch or clast == box_ch:
                box_idxs.append(i)
            if c1 == self.ne or clast == self.ne:
                angle_idxs.append(i)

        if len(box_idxs) == 3 and len(angle_idxs) == 3:
            boxcls = [outputs[i] for i in box_idxs]
            angles = [outputs[i] for i in angle_idxs]
            return boxcls, angles

        boxcls = outputs[:3]
        angles = outputs[3:]
        return boxcls, angles

    def decode(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxcls_lvls, angle_lvls = self._split_outputs(outputs)

        min_c = 4 * self.reg_max + self.nc

        def _stride_est_from_box(out: np.ndarray, min_channels: int) -> float:
            out_bchw = self._to_bchw(out, min_channels)
            _, _c, h, _w = out_bchw.shape
            return float(self.imgsz) / float(h)

        ordered_box = []
        ordered_ang = []
        for s in self.strides:
            best_box = None
            for idx, out in enumerate(boxcls_lvls):
                stride_est = _stride_est_from_box(out, min_c)
                diff = abs(stride_est - float(s))
                if best_box is None or diff < best_box[0]:
                    best_box = (diff, idx)
            assert best_box is not None
            ordered_box.append(boxcls_lvls[best_box[1]])

            best_ang = None
            for idx, out in enumerate(angle_lvls):
                stride_est = _stride_est_from_box(out, self.ne)
                diff = abs(stride_est - float(s))
                if best_ang is None or diff < best_ang[0]:
                    best_ang = (diff, idx)
            assert best_ang is not None
            ordered_ang.append(angle_lvls[best_ang[1]])

        boxcls_lvls = ordered_box
        angle_lvls = ordered_ang

        dist_all, cls_all, angle_all = [], [], []
        anchors_all, stride_all = [], []

        for i, boxcls in enumerate(boxcls_lvls):
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
            (xywh[keep_idx], conf[keep_idx, None], cls[keep_idx, None], angles[keep_idx]),
            axis=1,
        )

        sx = meta["w0"] / self.imgsz
        sy = meta["h0"] / self.imgsz
        det[:, 0] *= sx
        det[:, 1] *= sy
        det[:, 2] *= sx
        det[:, 3] *= sy

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


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def resolve_source_from_data(data_arg: str | None) -> Path:
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
    if not args.quant_meta:
        return

    import pickle

    with open(args.quant_meta, "rb") as f:
        cfg = pickle.load(f)
        no, stride, reg_max, nc, _dfl = cfg
        print(cfg)

    args.reg_max = 1
    args.nc = int(nc)

    if isinstance(stride, torch.Tensor):
        stride_vals = stride.detach().cpu().tolist()
    elif isinstance(stride, (list, tuple)):
        stride_vals = list(stride)
    else:
        stride_vals = [stride]

    args.strides = ",".join(str(int(s)) for s in stride_vals)
    print(
        f"Loaded quant meta from {args.quant_meta}: reg_max={args.reg_max}, nc={args.nc}, strides={args.strides}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv26 / YOLOv11 FPGA predictions.npz using the same decode + mAP.")
    parser.add_argument("--predictions-npz", type=str, required=True, help="Path to predictions.npz from FPGA inference.")
    parser.add_argument("--task", type=str, default="obb", choices=("detect", "obb"), help="Which head to evaluate (YOLOv26 / YOLOv11 detect or OBB).")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML for class names and validation images.")
    parser.add_argument("--quant-meta", type=str, default=None, help="Path to quantization metadata pkl (no/stride/reg_max/nc/dfl).")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size used on FPGA.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (rotated NMS for OBB).")
    parser.add_argument("--nc", type=int, default=1, help="Number of classes.")
    parser.add_argument("--reg-max", type=int, default=16, help="DFL reg_max.")
    parser.add_argument("--ne", type=int, default=1, help="Angle channels (OBB).")
    parser.add_argument("--strides", type=str, default="8,16,32", help="Comma-separated strides per output level.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--save-txt", action="store_true", help="Save detections as txt.")
    parser.add_argument("--no-save-vis", action="store_true", help="Do not save visualization images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.predictions_npz = str(Path(args.predictions_npz))

    if not os.path.exists(args.predictions_npz):
        raise FileNotFoundError(f"predictions.npz not found: {args.predictions_npz}")

    if args.task == "obb":
        save_default = "runs/npz_obb26_eval"
    else:
        save_default = "runs/npz_detect_eval"

    save_dir = Path(args.save_dir) if args.save_dir else Path(save_default)
    save_dir.mkdir(parents=True, exist_ok=True)

    apply_quant_meta(args)
    strides = [int(s.strip()) for s in args.strides.split(",") if s.strip()]
    if len(strides) != 3:
        raise ValueError(f"Expected 3 strides, got {strides}")

    source = resolve_source_from_data(args.data)

    if not source.exists():
        raise FileNotFoundError(f"Resolved image directory does not exist: {source}")

    print(f"[INFO] Using images directory: {source}")
    print(f"[INFO] Loading predictions: {args.predictions_npz}")
    data = np.load(args.predictions_npz)

    image_names = data["image_names"]
    output_shapes = data["output_shapes"]
    output_fixpoints = data["output_fixpoints"]
    num_outputs = int(data["num_outputs"])

    print(f"[INFO] predictions: {len(image_names)} images, {num_outputs} output tensors per image")
    print(f"[INFO] output_shapes: {output_shapes}")
    print(f"[INFO] output_fixpoints: {output_fixpoints}")

    if args.quant_meta is None:
        if "reg_max" in data:
            try:
                args.reg_max = int(np.array(data["reg_max"]).reshape(-1)[0])
                print(f"[INFO] Using reg_max from NPZ: {args.reg_max}")
            except Exception:
                pass

        if "strides" in data:
            try:
                strides = [int(x) for x in np.array(data["strides"]).reshape(-1).tolist()]
                print(f"[INFO] Using strides from NPZ: {strides}")
            except Exception:
                pass

    if args.task == "detect" and num_outputs != 3:
        raise ValueError(f"Detect task expects 3 outputs, but npz has num_outputs={num_outputs}")
    if args.task == "obb" and num_outputs != 6:
        raise ValueError(f"OBB task expects 6 outputs, but npz has num_outputs={num_outputs}")

    if args.task == "detect":
        decoder = DetectFromNPZ(
            data=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            nc=args.nc,
            reg_max=args.reg_max,
            strides=strides,
            max_det=args.max_det,
        )
        map_eval = DetectMapEvaluator(decoder.names)
    else:
        decoder = OBB26FromNPZ(
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
        map_eval = OBBMapEvaluator(decoder.names)

    total_det_time = 0.0
    total_images = 0
    total_skipped = 0

    t0 = time.time()
    warned_missing = 0
    for img_idx, img_name in enumerate(image_names):
        img_name_str = str(img_name)
        image_path = source / img_name_str
        if not image_path.exists():
            if warned_missing < 25:
                print(f"[WARN] image '{img_name_str}' not found under images directory; skipping")
            warned_missing += 1
            total_skipped += 1
            continue

        im0 = cv2.imread(str(image_path))
        if im0 is None:
            print(f"[WARN] Could not read image: {image_path}")
            total_skipped += 1
            continue

        meta = {"h0": int(im0.shape[0]), "w0": int(im0.shape[1])}

        pred_outputs: list[np.ndarray] = []
        for out_idx in range(num_outputs):
            key = f"pred_{img_idx}_output_{out_idx}"
            pred_int8 = data[key]
            fix_point = int(output_fixpoints[out_idx])
            scale = float(2**fix_point)
            pred_float = pred_int8.astype(np.float32) / scale
            pred_outputs.append(pred_float)

        start_det = time.time()
        if args.task == "detect":
            xyxy, scores = decoder.decode(pred_outputs)
            det = decoder.postprocess(xyxy, scores, meta)
        else:
            xywh, scores, angles = decoder.decode(pred_outputs)
            det = decoder.postprocess(xywh, scores, angles, meta)
        total_det_time += time.time() - start_det

        total_images += 1

        if not args.no_save_vis:
            vis = decoder.draw(im0, det)
            out_file = save_dir / img_name_str
            cv2.imwrite(str(out_file), vis)

        if args.save_txt:
            txt_file = save_dir / f"{Path(img_name_str).stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for row in det:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

        map_eval.update(image_path, det, im0.shape[:2])

        if (total_images % 10) == 0:
            avg = total_det_time / max(total_images, 1) * 1000
            print(f"[INFO] Processed {total_images} images (skipped {total_skipped}) | avg post+decode {avg:.2f}ms")

    elapsed = time.time() - t0
    print("")
    print("=" * 70)
    print("NPZ EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Images processed: {total_images}")
    print(f"Images skipped  : {total_skipped}")
    print(f"Total time      : {elapsed:.2f}s")
    if total_images > 0:
        print(f"Avg per image   : {total_det_time / total_images * 1000:.2f}ms")
    print("")

    if total_images == 0:
        print("[WARN] No images matched NPZ 'image_names' under the dataset val/test folder; mAP is zero.")
        results = {
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "fitness": 0.0,
        }
    else:
        results = map_eval.finalize()

    print("\n===== mAP Results =====")
    print(f"Precision:  {results.get('metrics/precision(B)', 0.0):.6f}")
    print(f"Recall:     {results.get('metrics/recall(B)', 0.0):.6f}")
    print(f"mAP50:      {results.get('metrics/mAP50(B)', 0.0):.6f}")
    print(f"mAP50-95:   {results.get('metrics/mAP50-95(B)', 0.0):.6f}")

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) if hasattr(v, "__float__") else v for k, v in results.items()}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

