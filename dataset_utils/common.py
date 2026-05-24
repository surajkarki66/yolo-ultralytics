"""
Shared helpers for dataset analysis and visualization.
"""

import logging
import pandas as pd
import yaml

from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
DATASET_SPLITS = ('train', 'valid', 'test')


def parse_class_names(names_raw: Union[dict, list, tuple, None]) -> Dict[int, str]:
    """Normalize class names (list or dict) to {id: name}."""
    if not names_raw:
        return {}
    if isinstance(names_raw, dict):
        return {int(k): str(v) for k, v in names_raw.items()}
    if isinstance(names_raw, (list, tuple)):
        return {i: str(name) for i, name in enumerate(names_raw)}


def load_class_names(data_dir: Path) -> Dict[int, str]:
    """Load class id -> name mapping from data.yaml if present."""
    yaml_path = Path(data_dir) / "data.yaml"
    if not yaml_path.exists():
        logger.warning("data.yaml not found. Proceeding without class names.")
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as yfile:
        data_config = yaml.safe_load(yfile) or {}
    return parse_class_names(data_config.get("names"))


def get_class_label(class_names: Dict[int, str], class_id: int) -> str:
    """Return human-readable label for a class id."""
    return class_names.get(class_id, f"Class {class_id}")


def list_image_paths(images_dir: Path) -> List[Path]:
    """List image files in a directory (case-insensitive extension match)."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def polygon_area_normalized(coords: List[float]) -> float:
    """Shoelace formula area for a normalized polygon (x,y pairs)."""
    n = len(coords) // 2
    if n < 3:
        return 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def objects_per_image(
    image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    split: Optional[str] = None,
) -> pd.Series:
    """
    Count annotations per image, including zero for images without boxes.

    Args:
        image_df: DataFrame with a 'filename' column.
        annotation_df: DataFrame with a 'filename' column (may be empty).
        split: If set, only include filenames starting with '{split}/'.
    """
    if image_df.empty:
        return pd.Series(dtype=int)

    filenames = image_df['filename']
    if split is not None:
        prefix = f"{split}/"
        filenames = filenames[filenames.str.startswith(prefix)]

    if annotation_df.empty:
        return pd.Series(0, index=filenames.values)

    ann_subset = annotation_df
    if split is not None:
        prefix = f"{split}/"
        ann_subset = annotation_df[annotation_df['filename'].str.startswith(prefix)]

    counts = ann_subset.groupby('filename').size() if not ann_subset.empty else pd.Series(dtype=int)
    return filenames.map(lambda name: int(counts.get(name, 0)))


def min_max_objects_per_image(
    image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    split: Optional[str] = None,
) -> tuple[int, int]:
    """Min and max object counts per image, treating unlabeled images as zero."""
    obj_counts = objects_per_image(image_df, annotation_df, split=split)
    if obj_counts.empty:
        return 0, 0
    return int(obj_counts.min()), int(obj_counts.max())
