"""
Module for performing exploratory data analysis on the dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from PIL import Image

from dataset_utils.common import (
    DATASET_SPLITS,
    SUPPORTED_IMAGE_EXTENSIONS,
    get_class_label,
    load_class_names,
    min_max_objects_per_image,
    polygon_area_normalized,
)

logger = logging.getLogger(__name__)


def _parse_label_line(parts: list[str], filename: str, label_path: Path, line: str) -> Optional[dict]:
    """Parse one YOLO bbox (5 fields) or OBB (9 fields) label line."""
    if len(parts) == 5:
        try:
            class_id, x_center, y_center, w, h = map(float, parts)
            return {
                'filename': filename,
                'class_id': int(class_id),
                'x_center': x_center,
                'y_center': y_center,
                'width': w,
                'height': h,
                'area': w * h,
                'format': 'bbox',
            }
        except ValueError:
            logger.warning("Invalid bbox label values in %s: %s", label_path, line)
            return None

    if len(parts) == 9:
        try:
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            return {
                'filename': filename,
                'class_id': class_id,
                'x_center': None,
                'y_center': None,
                'width': None,
                'height': None,
                'area': polygon_area_normalized(coords),
                'format': 'obb',
            }
        except ValueError:
            logger.warning("Invalid OBB label values in %s: %s", label_path, line)
            return None

    if parts:
        logger.warning(
            "Unsupported label line (%d fields) in %s: %s",
            len(parts),
            label_path,
            line,
        )
    return None


def perform_eda(data_dir: Path, output_dir: Path) -> None:
    """
    Perform exploratory data analysis on the dataset splits (train, valid, test).

    Args:
        data_dir: Path to the base dataset directory containing train, valid, test splits.
        output_dir: Path to save output files.
    """
    data_dir = Path(data_dir)
    class_names = load_class_names(data_dir)

    all_image_stats = []
    all_annotation_stats = []
    image_counts_by_split = {split: 0 for split in DATASET_SPLITS}

    for split in DATASET_SPLITS:
        split_images_dir = data_dir / split / 'images'
        split_labels_dir = data_dir / split / 'labels'

        if not split_images_dir.exists() or not split_labels_dir.exists():
            logger.warning("Skipping split '%s': required directories not found.", split)
            continue

        logger.info("Processing split: %s", split)

        for image_path in sorted(
            p for p in split_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ):
            label_path = split_labels_dir / f"{image_path.stem}.txt"
            image_counts_by_split[split] += 1
            filename = f"{split}/{image_path.name}"

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                aspect_ratio = (width / height) if height > 0 else float('nan')
                all_image_stats.append({
                    'filename': filename,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                })
            except Exception as e:
                logger.error("Image error %s: %s", image_path, e)
                continue

            if not label_path.exists():
                logger.warning(
                    "No label file for %s, counting image with zero annotations.",
                    image_path.name,
                )
                continue

            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        record = _parse_label_line(parts, filename, label_path, line.strip())
                        if record is not None:
                            all_annotation_stats.append(record)
            except Exception as e:
                logger.error("Label error %s: %s", label_path, e)

    image_df = pd.DataFrame(all_image_stats)
    annotation_df = pd.DataFrame(all_annotation_stats)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_obj, max_obj = min_max_objects_per_image(image_df, annotation_df)

    with open(output_dir / 'dataset_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Dataset Statistics (Across Splits)\n")
        f.write("=================================\n\n")

        f.write("Image Statistics:\n")
        f.write(f"Total images: {len(image_df)}\n")
        for split in DATASET_SPLITS:
            f.write(f"  {split.capitalize()}: {image_counts_by_split[split]}\n")

        if not image_df.empty:
            f.write(f"Average size: {image_df['width'].mean():.1f}x{image_df['height'].mean():.1f}\n")
            f.write(f"Width range: {image_df['width'].min()}–{image_df['width'].max()}\n")
            f.write(f"Height range: {image_df['height'].min()}–{image_df['height'].max()}\n")
            f.write(f"Avg aspect ratio: {image_df['aspect_ratio'].mean():.2f}\n")
            f.write(
                f"Aspect ratio range: {image_df['aspect_ratio'].min():.2f}"
                f"–{image_df['aspect_ratio'].max():.2f}\n\n"
            )
        else:
            f.write("No image data collected.\n\n")

        f.write("Annotation Statistics:\n")
        f.write(f"Total annotations: {len(annotation_df)}\n")
        if not image_df.empty:
            f.write(f"Avg objects per image: {len(annotation_df) / len(image_df):.2f}\n")
        else:
            f.write("Cannot compute avg objects per image.\n")

        if not annotation_df.empty:
            if 'format' in annotation_df.columns:
                format_counts = annotation_df['format'].value_counts()
                f.write("\nAnnotation formats:\n")
                for fmt, count in format_counts.items():
                    f.write(f"  {fmt}: {count}\n")

            class_counts = annotation_df['class_id'].value_counts().sort_index()
            f.write("\nClass Distribution:\n")
            for class_id, count in class_counts.items():
                class_label = get_class_label(class_names, int(class_id))
                logger.info("[Overall] %s: %s", class_label, count)
                f.write(f"{class_label}: {count}\n")
        else:
            f.write("No annotation data.\n")

        f.write("\nObjects per Image (Overall):\n")
        f.write(f"Min: {min_obj}\n")
        f.write(f"Max: {max_obj}\n\n")
        f.write("Split-wise Statistics\n=====================\n")

    for split in DATASET_SPLITS:
        split_img_df = image_df[image_df['filename'].str.startswith(f'{split}/')]
        split_ann_df = annotation_df[annotation_df['filename'].str.startswith(f'{split}/')]
        split_min_obj, split_max_obj = min_max_objects_per_image(
            image_df, annotation_df, split=split
        )

        with open(output_dir / 'dataset_statistics.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n-- {split.capitalize()} Split --\n")
            f.write(f"Images: {len(split_img_df)}\n")
            f.write(f"Annotations: {len(split_ann_df)}\n")

            if not split_img_df.empty:
                f.write(f"Avg objects/image: {len(split_ann_df) / len(split_img_df):.2f}\n")
                f.write(f"Avg size: {split_img_df['width'].mean():.1f}x{split_img_df['height'].mean():.1f}\n")
                f.write(f"Width range: {split_img_df['width'].min()}–{split_img_df['width'].max()}\n")
                f.write(f"Height range: {split_img_df['height'].min()}–{split_img_df['height'].max()}\n")
                f.write(f"Aspect ratio avg: {split_img_df['aspect_ratio'].mean():.2f}\n")
            else:
                f.write("No image data.\n")

            if not split_ann_df.empty:
                class_counts_split = split_ann_df['class_id'].value_counts().sort_index()
                f.write("  Class Distribution:\n")
                for class_id, count in class_counts_split.items():
                    class_label = get_class_label(class_names, int(class_id))
                    logger.info("[%s] %s: %s", split.capitalize(), class_label, count)
                    f.write(f"    {class_label}: {count}\n")

                f.write("  Objects per Image:\n")
                f.write(f"    Min: {split_min_obj}\n")
                f.write(f"    Max: {split_max_obj}\n")
            else:
                f.write("No annotation data.\n")
                if not split_img_df.empty:
                    f.write("  Objects per Image:\n")
                    f.write(f"    Min: {split_min_obj}\n")
                    f.write(f"    Max: {split_max_obj}\n")

    if not annotation_df.empty and 'class_id' in annotation_df.columns:
        order = sorted(annotation_df['class_id'].unique())
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='class_id', data=annotation_df, order=order)
        ax.set_xticklabels([get_class_label(class_names, int(cid)) for cid in order])
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution (All Splits)')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png')
        plt.close()
    else:
        logger.warning("No annotation data for plotting class distribution.")

    logger.info("EDA completed. Results saved.")
