"""Metrics, NMS, YAML, and IoU helpers for Vitis AI eval scripts (PyTorch + NumPy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class SimpleClass:
    """Debug-friendly repr listing public non-callable attributes."""

    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class YAML:
    @staticmethod
    def load(file: str | Path) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from e

        path = Path(file)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise AssertionError(f"Not a YAML file: {file}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f) or {}
        return data


def check_yaml(file: str | Path, suffix: tuple[str, ...] = (".yaml", ".yml"), hard: bool = True) -> str:
    p = Path(file).expanduser()
    if p.suffix.lower() not in suffix:
        if hard:
            raise FileNotFoundError(f"Expected YAML file with suffix {suffix}, got: {file}")
        return ""

    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    if hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    return ""


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: list[float] | np.ndarray, precision: list[float] | np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    if hasattr(np, "trapezoid"):
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    else:
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple:
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


class Metric(SimpleClass):
    def __init__(self) -> None:
        self.p, self.r, self.f1 = [], [], []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def mp(self) -> float:
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self) -> list[float]:
        return [self.mp, self.mr, self.map50, self.map]

    def fitness(self) -> float:
        w = [0.0, 0.0, 0.0, 1.0]
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    def update(self, results: tuple) -> None:
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results


class DetMetrics(SimpleClass):
    def __init__(self, names: dict[int, str] = {}) -> None:
        self.names = names
        self.box = Metric()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def update_stats(self, stat: dict[str, Any]) -> None:
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self) -> dict[str, np.ndarray]:
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        return stats

    @property
    def keys(self) -> list[str]:
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        return self.box.mean_results()

    @property
    def fitness(self) -> float:
        return self.box.fitness()

    @property
    def results_dict(self) -> dict[str, float]:
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))


class OBBMetrics(DetMetrics):
    pass


class TorchNMS:
    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        iou_func=box_iou,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes_sorted = boxes[sorted_idx]
        ious = iou_func(boxes_sorted, boxes_sorted)
        ious = ious.triu_(diagonal=1)
        pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(0, descending=True)

        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            if inter.sum() == 0:
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]
