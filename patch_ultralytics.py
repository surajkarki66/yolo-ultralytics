import os
import shutil
import ultralytics

# Get ultralytics package path dynamically
ultra_path = os.path.dirname(ultralytics.__file__)
print(f"[INFO] Ultralytics package located at: {ultra_path}")

# Source files with your custom modules
SRC_CONV = "./custom_layers/conv.py"
SRC_BLOCK = "./custom_layers/block.py"

# Destination folder: ultralytics nn/modules
DST_DIR = os.path.join(ultra_path, "nn", "modules")
DST_CONV = os.path.join(DST_DIR, "conv.py")
DST_BLOCK = os.path.join(DST_DIR, "block.py")

# Check if source files exist
if not os.path.isfile(SRC_CONV):
    print(f"[ERROR] Source file {SRC_CONV} does not exist!")
    exit(1)

if not os.path.isfile(SRC_BLOCK):
    print(f"[ERROR] Source file {SRC_BLOCK} does not exist!")
    exit(1)

# Check if destination directory exists
if not os.path.isdir(DST_DIR):
    print(f"[ERROR] Destination folder {DST_DIR} does not exist!")
    exit(1)

# Copy the custom conv.py file
try:
    shutil.copy2(SRC_CONV, DST_CONV)
    print(f"[OK] Successfully replaced conv.py at {DST_CONV}")
except Exception as e:
    print(f"[ERROR] Failed to copy conv.py: {e}")
    exit(1)

# Copy the custom block.py file
try:
    shutil.copy2(SRC_BLOCK, DST_BLOCK)
    print(f"[OK] Successfully replaced block.py at {DST_BLOCK}")
except Exception as e:
    print(f"[ERROR] Failed to copy block.py: {e}")
    exit(1)

print("[DONE] File replacement completed!")
