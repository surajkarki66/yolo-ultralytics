import os
import shutil
import ultralytics

# Get ultralytics package path dynamically
ultra_path = os.path.dirname(ultralytics.__file__)
print(f"[INFO] Ultralytics package located at: {ultra_path}")

# Source folder with your custom layers
SRC_DIR = "./custom_layers"

# Destination folder: ultralytics nn/modules
DST_DIR = os.path.join(ultra_path, "nn", "modules")

# Check if source exists
if not os.path.isdir(SRC_DIR):
    print(f"[ERROR] Source folder {SRC_DIR} does not exist!")
    exit(1)

# Check if destination exists
if not os.path.isdir(DST_DIR):
    print(f"[ERROR] Destination folder {DST_DIR} does not exist!")
    exit(1)

# Copy files (overwrite if exists)
for file_name in os.listdir(SRC_DIR):
    src_file = os.path.join(SRC_DIR, file_name)
    dst_file = os.path.join(DST_DIR, file_name)

    if os.path.isfile(src_file):
        shutil.copy2(src_file, dst_file)
        print(f"[OK] Copied {file_name} to {DST_DIR}")

print("[DONE] All files copied successfully!")
