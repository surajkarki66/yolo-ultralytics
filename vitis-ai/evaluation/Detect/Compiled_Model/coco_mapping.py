from __future__ import annotations

import json
import os

from functools import lru_cache
from typing import Dict, List, Optional, Tuple


@lru_cache(maxsize=8)
def _cached_gt_payload(
    gt_path: str,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[dict, ...], frozenset]:
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    categories: List[dict] = list(data.get("categories") or [])
    yolo_to_coco: Dict[int, int] = {}
    for c in categories:
        name = str(c.get("name", ""))
        if name.isdigit():
            yolo_to_coco[int(name)] = int(c["id"])
    items = tuple(sorted(yolo_to_coco.items()))
    cat_tuple = tuple(categories)
    ann_ids = frozenset(
        int(a["category_id"])
        for a in data.get("annotations") or []
        if a.get("category_id") is not None
    )
    return items, cat_tuple, ann_ids


def _resolve_gt_path(gt_path: str | None) -> str:
    path = gt_path or os.path.join(os.path.dirname(__file__), "gt_eval.json")
    return os.path.abspath(path)


def coco_category_id_from_gt(
    gt_path: str | None,
    yolo_class: Optional[int],
) -> int:
    """
    Map a YOLO class index to COCO category_id using only gt_eval.json.

    Parameters
    ----------
    gt_path
        Path to gt_eval.json (default: ``Compiled_Model/gt_eval.json`` next to this file).
    yolo_class
        Class index from the model (0 … num_classes-1). Use ``None`` when each
        box has only ``x, y, w, h, confidence`` (no class index); resolution
        then follows the same GT structure (one category, or one id in
        annotations, or digit map at index 0).
    """
    path = _resolve_gt_path(gt_path)
    idx = 0 if yolo_class is None else int(yolo_class)

    items, categories, ann_ids = _cached_gt_payload(path)
    m = dict(items)

    if idx in m:
        return m[idx]

    if len(categories) == 1:
        return int(categories[0]["id"])

    if not m and len(ann_ids) == 1:
        return next(iter(ann_ids))

    # VOC / Roboflow COCO without digit names: map YOLO index k to the k-th
    # leaf category id. Drop id 0 when it is a parent (e.g. name "objects")
    # and never appears in annotations — common in Roboflow exports.
    all_ids = sorted({int(c["id"]) for c in categories})
    if all_ids and all_ids[0] == 0 and 0 not in ann_ids:
        leaf_ids = [i for i in all_ids if i != 0]
    else:
        leaf_ids = all_ids

    if idx < len(leaf_ids):
        return leaf_ids[idx]

    raise KeyError(
        f"No COCO category mapping for YOLO class index {idx} in {path}. "
        f"Digit map: {m!r}, categories: {len(categories)}, "
        f"leaf category ids: {leaf_ids!r}, "
        f"annotation category_ids: {sorted(ann_ids)!r}"
    )