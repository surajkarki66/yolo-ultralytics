import os
from pathlib import Path

IMG_FORMATS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
output_file = Path("./data/val_ids.txt")

# Prefer data/val/; fall back to data/ when images live there directly.
image_dir = Path("./data/val")
if not image_dir.is_dir() or not any(image_dir.iterdir()):
    image_dir = Path("./data")

image_files = sorted(
    p.name
    for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in IMG_FORMATS
)

with open(output_file, "w", encoding="utf-8") as f:
    for name in image_files:
        f.write(name + "\n")

print(f"Saved {len(image_files)} image names from {image_dir} to {output_file}")
