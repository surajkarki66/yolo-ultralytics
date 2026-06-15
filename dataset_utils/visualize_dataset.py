"""
Visualize sample images with YOLO bbox and OBB annotations.
"""

import logging
import random
import cv2
import numpy as np
import supervision as sv

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from dataset_utils.common import (
    DATASET_SPLITS,
    get_class_label,
    list_image_paths,
    load_class_names,
)

logger = logging.getLogger(__name__)

OBB_COLOR = (0, 255, 0)
OBB_THICKNESS = 2


@dataclass
class ObbedAnnotation:
    """Oriented bounding box in pixel coordinates."""
    class_id: int
    points: np.ndarray  # shape (4, 2), int32


class SampleImageSaver:
    def __init__(
        self,
        base_dir: Path,
        output_dir: Path,
        num_samples: int = 5,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the SampleImageSaver.

        Args:
            base_dir: Base directory containing the dataset.
            output_dir: Directory to save output visualizations.
            num_samples: Number of samples to save from each split.
            class_names: Optional mapping of class id to display name.
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.class_names = class_names if class_names is not None else load_class_names(self.base_dir)
        self.subdirs = list(DATASET_SPLITS)

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving sample images to '%s'", self.output_dir)

    def get_random_samples(self, subdir: str) -> List[Path]:
        """Get random sample images from a split directory."""
        images_dir = self.base_dir / subdir / 'images'
        labels_dir = self.base_dir / subdir / 'labels'

        if not images_dir.is_dir() or not labels_dir.is_dir():
            logger.warning(
                "%s or %s not found. Skipping %s.",
                images_dir,
                labels_dir,
                subdir,
            )
            return []

        image_files = list_image_paths(images_dir)
        if len(image_files) > self.num_samples:
            return random.sample(image_files, self.num_samples)
        return image_files

    def read_annotations(
        self,
        label_path: Path,
        img_width: int,
        img_height: int,
    ) -> Tuple[Optional[sv.Detections], List[ObbedAnnotation]]:
        """Read YOLO bbox and OBB labels; return supervision Detections and OBB polygons."""
        detections_list: List[list] = []
        obb_list: List[ObbedAnnotation] = []

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue

                    if len(parts) == 5:
                        try:
                            class_id, x_center, y_center, w, h = map(float, parts)
                        except ValueError:
                            logger.warning(
                                "Invalid bbox label values in %s: %s",
                                label_path,
                                line.strip(),
                            )
                            continue

                        x1 = int((x_center - w / 2) * img_width)
                        y1 = int((y_center - h / 2) * img_height)
                        x2 = int((x_center + w / 2) * img_width)
                        y2 = int((y_center + h / 2) * img_height)

                        x1 = max(0, min(x1, img_width - 1))
                        y1 = max(0, min(y1, img_height - 1))
                        x2 = max(0, min(x2, img_width - 1))
                        y2 = max(0, min(y2, img_height - 1))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        detections_list.append([x1, y1, x2, y2, int(class_id)])
                        continue

                    if len(parts) == 9:
                        try:
                            class_id = int(float(parts[0]))
                            coords = list(map(float, parts[1:]))
                        except ValueError:
                            logger.warning(
                                "Invalid OBB label values in %s: %s",
                                label_path,
                                line.strip(),
                            )
                            continue

                        points = np.array(
                            [
                                [int(coords[i] * img_width), int(coords[i + 1] * img_height)]
                                for i in range(0, 8, 2)
                            ],
                            dtype=np.int32,
                        )
                        obb_list.append(ObbedAnnotation(class_id=class_id, points=points))
                        continue

                    logger.warning(
                        "Unsupported label line (%d fields) in %s: %s",
                        len(parts),
                        label_path,
                        line.strip(),
                    )

        except Exception as e:
            logger.error("Error processing annotations for %s: %s", label_path, e)
            return None, []

        detections_sv = None
        if detections_list:
            detections_np = np.array(detections_list)
            detections_sv = sv.Detections(
                xyxy=detections_np[:, :4],
                confidence=np.array([1.0] * len(detections_list)),
                class_id=detections_np[:, 4].astype(int),
            )

        return detections_sv, obb_list

    def _draw_obb(self, img: np.ndarray, obb_list: List[ObbedAnnotation]) -> np.ndarray:
        """Draw oriented bounding boxes on the image."""
        for obb in obb_list:
            cv2.polylines(
                img,
                [obb.points],
                isClosed=True,
                color=OBB_COLOR,
                thickness=OBB_THICKNESS,
            )
        return img

    def _detection_labels(self, detections: sv.Detections) -> List[str]:
        if detections.class_id is None:
            return []
        return [get_class_label(self.class_names, int(cid)) for cid in detections.class_id]

    def draw_annotations(self, image_path: Path, label_path: Path) -> Optional[np.ndarray]:
        """Draw bounding box and OBB annotations on the image."""
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("Error reading image: %s", image_path)
            return None

        height, width = img.shape[:2]
        detections, obb_list = self.read_annotations(label_path, width, height)

        has_bbox = detections is not None and len(detections) > 0
        has_obb = len(obb_list) > 0

        if not has_bbox and not has_obb:
            logger.warning("No valid annotations found for %s.", image_path.name)
            return img.copy()

        annotated_img = img.copy()

        if has_bbox:
            annotated_img = self.box_annotator.annotate(
                scene=annotated_img,
                detections=detections,
            )
            labels = self._detection_labels(detections)
            if labels:
                annotated_img = self.label_annotator.annotate(
                    scene=annotated_img,
                    detections=detections,
                    labels=labels,
                )

        if has_obb:
            annotated_img = self._draw_obb(annotated_img, obb_list)

        return annotated_img

    def save_samples(self) -> None:
        """Save annotated sample images from each split."""
        for subdir in self.subdirs:
            logger.info("Processing %s set...", subdir)

            set_output_dir = self.output_dir / subdir
            set_output_dir.mkdir(exist_ok=True)

            sample_images = self.get_random_samples(subdir)
            if not sample_images:
                logger.warning("No images found in %s. Skipping.", self.base_dir / subdir)
                continue

            for img_path in tqdm(sample_images, desc=f"Saving {subdir} samples"):
                label_path = self.base_dir / subdir / 'labels' / f"{img_path.stem}.txt"

                if not label_path.exists():
                    logger.warning(
                        "No label file found for %s in %s",
                        img_path.name,
                        self.base_dir / subdir / 'labels',
                    )
                    continue

                annotated_img = self.draw_annotations(img_path, label_path)
                if annotated_img is not None:
                    output_path = set_output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                    cv2.imwrite(str(output_path), annotated_img)


def visualize_sample_annotated_dataset(
    data_dir: Path,
    output_dir: Path,
    num_samples: int = 10,
    classes: Optional[Dict[int, str]] = None,
) -> None:
    """
    Visualize a sample of annotated images from the dataset.

    Args:
        data_dir: Path to the dataset directory.
        output_dir: Path to save output visualizations.
        num_samples: Number of samples to visualize per split.
        classes: Optional mapping of class id to name; loads from data.yaml if omitted.
    """
    saver = SampleImageSaver(
        base_dir=data_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        class_names=classes,
    )
    saver.save_samples()
