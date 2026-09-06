import argparse
import os
import sys
import cv2

from utils import util


def draw_obb_predictions(im, pred_np, names, input_size, conf_threshold=0.0):
    """
    Draw OBB predictions on image. pred_np: (k, 7) [x, y, w, h, conf, cls, angle] in input_size space.
    Resizes im to (input_size, input_size) so coordinates match.
    """
    im = cv2.resize(im, (input_size, input_size))
    for i in range(len(pred_np)):
        pred = pred_np[i]
        conf = float(pred[4])
        if conf < conf_threshold:
            continue
        xywhr = [float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]), float(pred[6])]
        cls_id = int(pred[5])
        label = f"{names[cls_id] if cls_id < len(names) else cls_id} {conf:.2f}"
        util.draw_rotated_box(im, xywhr, cls_id, label)
    return im


def main():
    parser = argparse.ArgumentParser(
        description="OBB inference from .npz: post-process and save annotated images"
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to .npz from fpga_inference.py")
    parser.add_argument("--config", type=str, required=True, help="Path to OBB model config .pkl")
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images (same as used for fpga_inference.py, e.g. test_data/test/images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/inference_obb",
        help="Directory to save annotated images",
    )
    parser.add_argument("--input-size", type=int, default=416, help="Model input size (square)")
    parser.add_argument(
        "--nc",
        type=int,
        default=None,
        metavar="N",
        help="Override class count for NPZ post-process (default: from --config .pkl)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.0,
        help="Min confidence to draw (0 = draw all after NMS)",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Class names (default: 0, 1, 2, ...)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.npz):
        print(f"NPZ not found: {args.npz}")
        sys.exit(1)
    if not os.path.isfile(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)
    if not os.path.isdir(args.images_dir):
        print(f"Images dir not found: {args.images_dir}")
        sys.exit(1)

    # Reuse loader from evaluate_obb (fix_point dequantization + OBB post-process)
    from evaluate_obb import load_predictions_from_npz

    print(f"Loading predictions from {args.npz} (fix_point dequantization)...")
    name_to_pred = load_predictions_from_npz(
        args.npz,
        args.config,
        args.iou_threshold,
        num_detect=3,
        num_angle=3,
        nc=args.nc,
    )
    print(f"Got predictions for {len(name_to_pred)} images.")

    names = args.names if args.names else [str(i) for i in range(20)]
    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    missing = 0
    for basename, pred_np in name_to_pred.items():
        img_path = os.path.join(args.images_dir, basename)
        if not os.path.isfile(img_path):
            missing += 1
            if missing <= 5:
                print(f"  Skip (image not found): {img_path}")
            continue
        im = cv2.imread(img_path)
        if im is None:
            missing += 1
            continue
        im_out = draw_obb_predictions(
            im, pred_np, names, args.input_size, conf_threshold=args.conf_threshold
        )
        out_path = os.path.join(args.output_dir, basename)
        cv2.imwrite(out_path, im_out)
        saved += 1
        if saved % 100 == 0:
            print(f"  Saved {saved} annotated images...")

    print(f"Done. Saved {saved} images to {args.output_dir}")
    if missing:
        print(f"  Skipped {missing} images (not found under {args.images_dir})")


if __name__ == "__main__":
    main()
