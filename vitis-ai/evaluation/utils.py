from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

class SimpleClass:
    """A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings, showing
    all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

class DataExportMixin:
    """Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Polars
    DataFrame, CSV, and JSON.

    Methods:
        to_df: Convert summary to a Polars DataFrame.
        to_csv: Export results as a CSV string.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """Create a Polars DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (polars.DataFrame): Polars DataFrame containing the summary data.
        """
        import polars as pl

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """Export results or metrics to CSV string format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): CSV content as string.
        """
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)

        try:
            return df.write_csv()
        except Exception:
            # Minimal string conversion for any remaining complex types
            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): JSON-formatted string of the results.
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class DetMetrics(SimpleClass, DataExportMixin):
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        update_stats: Update statistics by appending new values to existing stat collections.
        process: Process predicted results for object detection and update metrics.
        clear_stats: Clear the stored statistics.
        keys: Return a list of keys for accessing specific metrics.
        mean_results: Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95.
        class_result: Return the result of evaluating the performance of an object detection model on a specific class.
        maps: Return mean Average Precision (mAP) scores per class.
        fitness: Return the fitness of box object.
        ap_class_index: Return the average precision index per class.
        results_dict: Return dictionary of computed performance metrics and statistics.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Return a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """Initialize a DetMetrics instance with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, Any]): Dictionary containing new statistical values to append. Keys should match existing
                keys in self.stats.
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Process predicted results for object detection and update metrics.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # to numpy
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
        return stats

    def clear_stats(self):
        """Clear the stored statistics."""
        for v in self.stats.values():
            v.clear()

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self) -> list[float]:
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """Return the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """Return the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Return dictionary of computed performance metrics and statistics."""
        keys = [*self.keys, "fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes
        shared scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
            normalize (bool): For Detect metrics, everything is normalized by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
                values.

        Examples:
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {
            "Box-P": self.box.p,
            "Box-R": self.box.r,
            "Box-F1": self.box.f1,
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]

def xyxyxyxy2xywhr(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

    Args:
        x (np.ndarray | torch.Tensor): Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.

    Returns:
        (np.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation
            values are in radians from [-pi/4, 3pi/4).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        # convert angle to radian and normalize to [-pi/4, 3pi/4)
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

class YAML:
    @staticmethod
    def load(file: str | Path, append_filename: bool = False) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required. Install it with: pip install pyyaml") from e

        path = Path(file)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise AssertionError(f"Not a YAML file: {file}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = yaml.safe_load(f) or {}
        if append_filename and isinstance(data, dict):
            data["yaml_file"] = str(path)
        return data


def check_yaml(file: str | Path, suffix: tuple[str, ...] = (".yaml", ".yml"), hard: bool = True) -> str:
    """Resolve a YAML path."""
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
    """Pairwise IoU for xyxy boxes."""
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
    """Pairwise probabilistic IoU for oriented boxes in xywhr format."""
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
    """Box filter smoothing used for F1 curve peak selection."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall: list[float] | np.ndarray, precision: list[float] | np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute average precision under precision-recall curve."""
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
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute AP metrics per class."""
    _ = (plot, on_plot, save_dir, names, prefix)

    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = np.linspace(0, 1, 1000), []
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
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """Storage and helpers for AP metrics."""

    def __init__(self) -> None:
        self.p, self.r, self.f1 = [], [], []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self):
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        return self.all_ap.mean(1) if len(self.all_ap) else []

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

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self) -> np.ndarray:
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self) -> float:
        w = [0.0, 0.0, 0.0, 1.0]
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    def update(self, results: tuple) -> None:
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self) -> list:
        return []

    @property
    def curves_results(self) -> list[list]:
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass):
    """Detection metrics interface for mAP reporting."""

    def __init__(self, names: dict[int, str] = {}) -> None:
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.nt_per_class = None
        self.nt_per_image = None

    def update_stats(self, stat: dict[str, Any]) -> None:
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        _ = save_dir
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
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
    """Oriented detection metrics"""


class TorchNMS:
    """Minimal NMS helpers used by this workspace."""

    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ) -> torch.Tensor:
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes_sorted = boxes[sorted_idx]
        ious = iou_func(boxes_sorted, boxes_sorted)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes_sorted.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            scores_ = scores[sorted_idx].clone()
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            pick = torch.topk(scores_, scores_.shape[0]).indices
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