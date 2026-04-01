import os

# Set your image directory path here
image_dir = './data/val/'
output_file = './data/val_ids.txt'
valid_exts = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm'}

# Get list of image files only
image_files = sorted(
    f
    for f in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
)

# Write full paths to val_ids.txt
with open(output_file, 'w') as f:
    for img in image_files:
        full_path = os.path.abspath(os.path.join(image_dir, img))
        f.write(full_path + '\n')

print(f"Saved {len(image_files)} file paths to {output_file}")
