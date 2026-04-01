import os
import shutil
import ultralytics

def main():
    # Get ultralytics package path dynamically
    ultra_path = os.path.dirname(ultralytics.__file__)
    print(f"[INFO] Ultralytics package located at: {ultra_path}")

    # Source folder with your custom layers
    src_dir = "./custom_layers"

    # Destination folder: ultralytics nn/modules
    dst_dir = os.path.join(ultra_path, "nn", "modules")

    # Check if source exists
    if not os.path.isdir(src_dir):
        print(f"[ERROR] Source folder {src_dir} does not exist!")
        raise SystemExit(1)

    # Check if destination exists
    if not os.path.isdir(dst_dir):
        print(f"[ERROR] Destination folder {dst_dir} does not exist!")
        raise SystemExit(1)

    # Copy files (overwrite if exists)
    for file_name in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"[OK] Copied {file_name} to {dst_dir}")

    print("[DONE] All files copied successfully!")


if __name__ == "__main__":
    main()