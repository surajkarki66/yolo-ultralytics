import logging
import shutil
import pandas as pd
import yaml

from collections import Counter
from pathlib import Path
from sklearn.model_selection import KFold
from ultralytics import YOLO

from dataset_utils.common import SUPPORTED_IMAGE_EXTENSIONS, parse_class_names
from .utils import drop_keys, load_config, require_file

logger = logging.getLogger(__name__)

YOLO_PARAM_EXCLUDE = frozenset({
    "dataset_path",
    "data_yaml_path",
    "k_splits",
    "random_state",
    "task",
    "model",
})


def _list_images(images_dir: Path) -> dict[str, Path]:
    """Map image stem to path for all supported extensions."""
    images: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            logger.warning(
                "Duplicate image stem '%s'; keeping %s",
                path.stem,
                images[path.stem],
            )
            continue
        images[path.stem] = path
    return images


def _copy_split_files(source_path: Path, target_images: Path, target_labels: Path) -> None:
    """Merge train/valid splits into flat image/label folders."""
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        for dtype, target_dir in (("images", target_images), ("labels", target_labels)):
            src_dir = source_path / split / dtype
            if not src_dir.is_dir():
                logger.warning("Missing split directory: %s", src_dir)
                continue
            for src_file in src_dir.iterdir():
                if not src_file.is_file():
                    continue
                dest = target_dir / src_file.name
                if dest.exists():
                    logger.warning("Skipping duplicate file: %s", dest.name)
                    continue
                shutil.copy(src_file, dest)


def _parse_label_file(label_path: Path) -> Counter:
    """Parse YOLO label file into class-id counts."""
    counter: Counter = Counter()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
                counter[cls_id] += 1
            except (ValueError, IndexError):
                logger.warning("Invalid label line in %s: %s", label_path, line.strip())
    return counter


def _names_for_yaml(class_names: dict[int, str]) -> list[str]:
    """Convert class map to ordered list for Ultralytics YAML."""
    if not class_names:
        return []
    return [class_names[class_id] for class_id in sorted(class_names)]


def run_cross_validation():
    """Run k-fold cross-validation and return the summary DataFrame."""
    cv_cfg = load_config("cross_validation")

    source_path = Path(cv_cfg.get("dataset_path", "./dataset/hpd3_320/yolov8"))
    target_images_path = source_path / "images"
    target_labels_path = source_path / "labels"
    yaml_file = Path(cv_cfg.get("data_yaml_path", source_path / "data.yaml"))
    ksplit = cv_cfg.get("k_splits", 10)
    random_state = cv_cfg.get("random_state", 42)
    model_path = cv_cfg.get("model", "yolov8n.pt")

    require_file(yaml_file, "Dataset yaml")
    _copy_split_files(source_path, target_images_path, target_labels_path)

    with open(yaml_file, "r", encoding="utf-8") as yfile:
        data_yaml = yaml.safe_load(yfile) or {}
    class_names_map = parse_class_names(data_yaml.get("names"))

    label_files = sorted(target_labels_path.glob("*.txt"))
    if not label_files:
        raise ValueError(f"No label files found in {target_labels_path}")

    parsed_labels = {label_file.stem: _parse_label_file(label_file) for label_file in label_files}
    label_class_ids = set()
    for counter in parsed_labels.values():
        label_class_ids.update(counter.keys())

    class_ids = sorted(set(class_names_map.keys()) | label_class_ids)
    if not class_ids:
        raise ValueError("No classes found in data.yaml or label files")

    labels_df = pd.DataFrame(index=sorted(parsed_labels.keys()), columns=class_ids, dtype=float)
    for stem, counter in parsed_labels.items():
        for cls_id, count in counter.items():
            labels_df.at[stem, cls_id] = count
    labels_df = labels_df.fillna(0.0)

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=random_state)
    kfolds = list(kf.split(labels_df))

    fold_names = [f"split_{i + 1}" for i in range(ksplit)]
    label_distribution = pd.DataFrame(index=fold_names, columns=class_ids)
    for i, (train_idx, val_idx) in enumerate(kfolds):
        train_sum = labels_df.iloc[train_idx].sum()
        val_sum = labels_df.iloc[val_idx].sum()
        label_distribution.loc[f"split_{i + 1}"] = val_sum / (train_sum + 1e-7)

    label_distribution.to_csv(source_path / "kfold_label_distribution.csv")

    kset_path = source_path / "kfold"
    shutil.rmtree(kset_path, ignore_errors=True)
    kset_path.mkdir(parents=True, exist_ok=True)

    all_images = _list_images(target_images_path)
    yaml_paths: list[str] = []
    names_list = _names_for_yaml(class_names_map)

    for i, (train_idx, val_idx) in enumerate(kfolds):
        train_files = labels_df.iloc[train_idx].index
        val_files = labels_df.iloc[val_idx].index

        train_txt = kset_path / f"train_{i}.txt"
        val_txt = kset_path / f"val_{i}.txt"

        with open(train_txt, "w", encoding="utf-8") as f:
            for name in train_files:
                if name in all_images:
                    f.write(f"{all_images[name]}\n")

        with open(val_txt, "w", encoding="utf-8") as f:
            for name in val_files:
                if name in all_images:
                    f.write(f"{all_images[name]}\n")

        yaml_data = {
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "names": names_list if names_list else class_names_map,
        }
        yaml_path = kset_path / f"data_fold_{i}.yaml"
        with open(yaml_path, "w", encoding="utf-8") as yf:
            yaml.safe_dump(yaml_data, yf)
        yaml_paths.append(str(yaml_path))

    yolo_params = drop_keys(cv_cfg, *YOLO_PARAM_EXCLUDE)
    yolo_params["project"] = cv_cfg.get("project", "kfold_result")

    results = []
    for i, yaml_path in enumerate(yaml_paths):
        logger.info("Training fold %d with %s", i + 1, yaml_path)
        model = YOLO(model_path)
        train_results = model.train(data=yaml_path, name=f"fold_{i}", **yolo_params)
        if train_results is None:
            logger.warning("Fold %d returned no training results", i + 1)
            continue
        results.append(train_results)

    if not results:
        raise RuntimeError("No metrics collected from cross-validation folds")

    metric_values: dict[str, list] = {}
    for result in results:
        results_dict = getattr(result, "results_dict", None)
        if not results_dict:
            logger.warning("Skipping fold result without results_dict")
            continue
        for metric, value in results_dict.items():
            metric_values.setdefault(metric, []).append(value)

    if not metric_values:
        raise RuntimeError("No metric values found in cross-validation results")

    metric_df = pd.DataFrame(metric_values)
    summary = metric_df.describe().loc[["mean", "std", "min", "max"]]
    summary.to_csv(source_path / "kfold_metrics_summary.csv")

    logger.info("K-fold training completed")
    logger.info("\n%s", summary)
    return summary


if __name__ == "__main__":
    run_cross_validation()
