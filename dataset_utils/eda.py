"""
Module for performing exploratory data analysis on the dataset.
"""

import logging
from pathlib import Path
import pandas as pd
# ✅ Use non-GUI backend to avoid Qt/xcb errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml

logger = logging.getLogger(__name__)

def perform_eda(data_dir: Path, output_dir: Path) -> None:
    """
    Perform exploratory data analysis on the dataset splits (train, valid, test).
    
    Args:
        data_dir (Path): Path to the base dataset directory containing train, valid, test splits.
        output_dir (Path): Path to save output files.
    """
    data_dir = Path(data_dir)
    
    # Load data.yaml for class names
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as yfile:
            data_config = yaml.safe_load(yfile)
            class_names = data_config.get("names", [])
    else:
        logger.warning("data.yaml not found. Proceeding without class names.")
        class_names = []

    all_image_stats = []
    all_annotation_stats = []
    dataset_splits = ['train', 'valid', 'test']
    image_counts_by_split = {split: 0 for split in dataset_splits}

    for split in dataset_splits:
        split_images_dir = data_dir / split / 'images'
        split_labels_dir = data_dir / split / 'labels'
        
        if not split_images_dir.exists() or not split_labels_dir.exists():
            logger.warning(f"Skipping split '{split}': Required directories not found.")
            continue
        
        logger.info(f"Processing split: {split}")

        for image_path in split_images_dir.glob('*.jpg'):
            label_path = split_labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                logger.warning(f"No label file for {image_path.name}, skipping.")
                continue
            
            image_counts_by_split[split] += 1

            try:
                img = Image.open(image_path)
                width, height = img.size
                all_image_stats.append({
                    'filename': f"{split}/{image_path.name}",
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height
                })
            except Exception as e:
                logger.error(f"Image error {image_path}: {e}")
                continue

            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id, x_center, y_center, w, h = map(float, parts)
                                all_annotation_stats.append({
                                    'filename': f"{split}/{image_path.name}",
                                    'class_id': int(class_id),
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': w,
                                    'height': h,
                                    'area': w * h
                                })
                            except ValueError:
                                logger.warning(f"Invalid label values in {label_path}: {line.strip()}")
            except Exception as e:
                logger.error(f"Label error {label_path}: {e}")

    image_df = pd.DataFrame(all_image_stats)
    annotation_df = pd.DataFrame(all_annotation_stats)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'dataset_statistics.txt', 'w') as f:
        f.write("Dataset Statistics (Across Splits)\n")
        f.write("=================================\n\n")

        f.write("Image Statistics:\n")
        f.write(f"Total images: {len(image_df)}\n")
        for split in dataset_splits:
            f.write(f"  {split.capitalize()}: {image_counts_by_split[split]}\n")

        if not image_df.empty:
            f.write(f"Average size: {image_df['width'].mean():.1f}x{image_df['height'].mean():.1f}\n")
            f.write(f"Width range: {image_df['width'].min()}–{image_df['width'].max()}\n")
            f.write(f"Height range: {image_df['height'].min()}–{image_df['height'].max()}\n")
            f.write(f"Avg aspect ratio: {image_df['aspect_ratio'].mean():.2f}\n")
            f.write(f"Aspect ratio range: {image_df['aspect_ratio'].min():.2f}–{image_df['aspect_ratio'].max():.2f}\n\n")
        else:
            f.write("No image data collected.\n\n")

        f.write("Annotation Statistics:\n")
        f.write(f"Total annotations: {len(annotation_df)}\n")
        if not image_df.empty:
            f.write(f"Avg objects per image: {len(annotation_df) / len(image_df):.2f}\n")
        else:
            f.write("Cannot compute avg objects per image.\n")

        if not annotation_df.empty:
            class_counts = annotation_df['class_id'].value_counts().sort_index()
            f.write("\nClass Distribution:\n")
            for class_id, count in class_counts.items():
                class_label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                print(f"[Overall] {class_label}: {count}")
                f.write(f"{class_label}: {count}\n")
        else:
            f.write("No annotation data.\n")

    # Min/max objects per image
    min_obj, max_obj = 0, 0
    if not annotation_df.empty and not image_df.empty:
        obj_per_img = annotation_df.groupby('filename').size()
        if not obj_per_img.empty:
            min_obj = obj_per_img.min()
            max_obj = obj_per_img.max()

    with open(output_dir / 'dataset_statistics.txt', 'a') as f:
        f.write("\nObjects per Image (Overall):\n")
        f.write(f"Min: {min_obj}\n")
        f.write(f"Max: {max_obj}\n\n")
        f.write("Split-wise Statistics\n=====================\n")

    # Split-wise stats
    for split in dataset_splits:
        split_img_df = image_df[image_df['filename'].str.startswith(f'{split}/')]
        split_ann_df = annotation_df[annotation_df['filename'].str.startswith(f'{split}/')]

        with open(output_dir / 'dataset_statistics.txt', 'a') as f:
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
                    class_label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    print(f"[{split.capitalize()}] {class_label}: {count}")
                    f.write(f"    {class_label}: {count}\n")

                objs_split = split_ann_df.groupby('filename').size()
                if not objs_split.empty:
                    f.write("  Objects per Image:\n")
                    f.write(f"    Min: {objs_split.min()}\n")
                    f.write(f"    Max: {objs_split.max()}\n")
            else:
                f.write("No annotation data.\n")

    # Plot class distribution
    if not annotation_df.empty and 'class_id' in annotation_df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='class_id', data=annotation_df)
        if class_names:
            plt.xticks(ticks=range(len(class_names)), labels=class_names)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution (All Splits)')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png')
        plt.close()
    else:
        logger.warning("No annotation data for plotting class distribution.")

    logger.info("EDA completed. Results saved.")
