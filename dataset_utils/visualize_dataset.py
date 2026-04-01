import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import supervision as sv
from dataclasses import dataclass

@dataclass
class Annotation:
    """Class to store annotation data"""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

class SampleImageSaver:
    def __init__(self, base_dir: Path, output_dir: Path, num_samples: int = 5):
        """
        Initialize the SampleImageSaver

        Args:
            base_dir (Path): Base directory containing the dataset
            output_dir (Path): Directory to save output visualizations
            num_samples (int): Number of samples to save from each set
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.subdirs = ['train', 'valid', 'test'] # Assuming these subdirectories exist
        self.image_extensions = {'.jpg', '.jpeg', '.png'}

        # Initialize supervision annotators with correct parameters
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving sample images to '{self.output_dir}'")

    def get_random_samples(self, subdir):
        """Get random sample images from a directory"""
        images_dir = self.base_dir / subdir / 'images'
        labels_dir = self.base_dir / subdir / 'labels'

        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"Warning: {images_dir} or {labels_dir} not found. Skipping {subdir}.")
            return []

        # Get all image files
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(list(images_dir.glob(f'*{ext}')))

        # Get random samples
        if len(image_files) > self.num_samples:
            return random.sample(image_files, self.num_samples)
        return image_files

    def read_annotations(self, label_path: Path, img_width: int, img_height: int):
        """Read annotations from YOLO format label file and convert to supervision Detections"""
        detections_list = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    # Parse YOLO format (class_id, x_center, y_center, width, height)
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, w, h = map(float, parts)

                        # Convert normalized coordinates to pixel coordinates (xyxy format)
                        x1 = int((x_center - w/2) * img_width)
                        y1 = int((y_center - h/2) * img_height)
                        x2 = int((x_center + w/2) * img_width)
                        y2 = int((y_center + h/2) * img_height)

                        # Append detection box coordinates and class_id
                        detections_list.append([x1, y1, x2, y2, int(class_id)])

        except Exception as e:
            print(f"Error processing annotations for {label_path}: {e}")
            return None

        if not detections_list:
            return None # Return None if no annotations were read

        # Convert list to numpy array for supervision Detections
        detections_np = np.array(detections_list)
        xyxy = detections_np[:, :4]
        class_ids = detections_np[:, 4].astype(int)

        # Create supervision Detections object
        detections_sv = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([1.0] * len(detections_list)),  # Assuming confidence of 1.0 for ground truth
            class_id=class_ids
        )

        return detections_sv

    def draw_annotations(self, image_path: Path, label_path: Path):
        """Draw bounding box annotations on the image using supervision"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error reading image: {image_path}")
            return None

        height, width = img.shape[:2]

        # Read and process annotations
        detections = self.read_annotations(label_path, width, height)

        if detections is None or len(detections) == 0:
             print(f"No valid annotations found for {image_path.name}. Skipping drawing.")
             # Return original image if no detections to draw
             return img.copy()


        # Draw annotations using supervision's BoxAnnotator

        annotated_img = self.box_annotator.annotate(
            scene=img.copy(),
            detections=detections,
        )

        return annotated_img

    def save_samples(self):
        """Save annotated sample images from each set"""
        # Assuming output_dir is already created in __init__

        for subdir in self.subdirs:
            print(f"\nProcessing {subdir} set...")

            # Create subdirectory for this set within the main output directory
            set_output_dir = self.output_dir / subdir
            set_output_dir.mkdir(exist_ok=True)

            # Get random samples
            sample_images = self.get_random_samples(subdir)

            if not sample_images:
                print(f"No images found in {self.base_dir / subdir}. Skipping.")
                continue

            # Process each sample
            for img_path in tqdm(sample_images, desc=f"Saving {subdir} samples"):
                # Get corresponding label file
                label_path = self.base_dir / subdir / 'labels' / f"{img_path.stem}.txt"

                if not label_path.exists():
                    print(f"Warning: No label file found for {img_path.name} in {self.base_dir / subdir / 'labels'}")
                    # Optionally save the image without annotations
                    # shutil.copy(img_path, set_output_dir)
                    continue

                # Draw annotations
                annotated_img = self.draw_annotations(img_path, label_path)

                if annotated_img is not None:
                    # Save the annotated image
                    output_path = set_output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                    cv2.imwrite(str(output_path), annotated_img)

def visualize_sample_annotated_dataset(
    data_dir: Path, output_dir: Path, num_samples: int = 10, classes: dict = None
) -> None:
    """
    Visualize a sample of annotated images from the dataset using the SampleImageSaver.

    Args:
        data_dir (Path): Path to the dataset directory.
        output_dir (Path): Path to save output visualizations.
        num_samples (int): Number of samples to visualize (default: 10).
        classes (dict): Optional dictionary mapping class IDs to names (not used by current SampleImageSaver).
    """
    # Create an instance of SampleImageSaver and run the saving process
    # Note: The SampleImageSaver expects the dataset structure like data_dir/train, data_dir/valid, data_dir/test
    # If your dataset structure is different, you might need to adjust SampleImageSaver accordingly.
    saver = SampleImageSaver(base_dir=data_dir, output_dir=output_dir, num_samples=num_samples)
    saver.save_samples() 