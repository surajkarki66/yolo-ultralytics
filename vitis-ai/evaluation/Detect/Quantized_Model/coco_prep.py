import argparse
import json
import os
import sys

_DETECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _DETECT_DIR not in sys.path:
    sys.path.insert(0, _DETECT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Convert output_boxes_*.txt to JSON for evaluate.py")
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to the boxes .txt produced by post_processing_onnx.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: coco-preped-<input_stem>.json in the current directory)",
    )
    parser.add_argument(
        "--gt",
        default="gt_eval.json",
        help="Must match the gt_eval.json you pass to evaluate.py.",
    )
    args = parser.parse_args()

    path = os.path.abspath(os.path.normpath(args.input))
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = args.output
    if out_path is None:
        out_path = f"coco-preped-{stem}.json"
    elif not os.path.isabs(out_path):
        out_path = os.path.join(os.getcwd(), out_path)

    gt_path = args.gt
    if not os.path.isabs(gt_path):
        gt_path = os.path.join(os.getcwd(), gt_path)
    merged_dict = {}
    current_image = None

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input boxes file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) >= 3 and parts[0] == "Image" and parts[1] == "Prediction:":
                current_image = parts[2]
                if current_image not in merged_dict:
                    merged_dict[current_image] = []
                continue

            if parts[0].startswith("Boxes"):
                continue

            if current_image is None:
                continue

            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue

            if len(nums) == 6:
                x, y, w, h, conf, yolo_cls = nums
                merged_dict[current_image].append([x, y, w, h, conf, int(yolo_cls)])
            elif len(nums) == 5:
                x, y, w, h, conf = nums
                merged_dict[current_image].append([x, y, w, h, conf])
            else:
                continue

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged_dict, f)

    n = sum(len(v) for v in merged_dict.values())
    print(f"Wrote {out_path} ({n} boxes across {len(merged_dict)} images)")
    print(f"Run evaluate.py with the same --gt / gt_eval path you used here: {os.path.abspath(gt_path)}")


if __name__ == "__main__":
    main()
