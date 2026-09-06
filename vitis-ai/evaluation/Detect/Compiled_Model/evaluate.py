import os
import sys
import json

_DETECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _DETECT_DIR not in sys.path:
    sys.path.insert(0, _DETECT_DIR)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from coco_mapping import coco_category_id_from_gt


def evaluate(dt_path, save_path="metrics.json", gt_path=None):
    # Load detection results
    if isinstance(dt_path, str):
        with open(dt_path) as f:
            dt = json.load(f)
    else:
        dt = dt_path

    if not isinstance(dt, dict):
        raise TypeError("Detection result format is not correct. Expected a dict keyed by filename.")

    gt_json = gt_path or os.path.join(os.path.dirname(__file__), "gt_eval.json")
    # Load COCO ground truth
    coco_gt = COCO(gt_json)
    filename_to_id = {img['file_name']: img['id'] for img in coco_gt.dataset['images']}

    # Convert results to COCO format
    dt_coco = []
    ids = []
    for filename, bboxes in dt.items():
        if filename not in filename_to_id:
            continue
        image_id = filename_to_id[filename]
        ids.append(image_id)
        for box in bboxes:
            bbox_xywh = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            score = float(box[4])
            yolo_cls = int(box[5]) if len(box) >= 6 else None
            category_id = coco_category_id_from_gt(gt_json, yolo_cls)
            dt_coco.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox_xywh,
                "score": score,
            })

    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(dt_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = ids
    coco_eval.params.maxDets = [3, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save all metrics exactly as they appear in the summary
    metrics = {
        "AP_all_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[0]),
        "AP_all_IoU_0.50_maxDets_100": float(coco_eval.stats[1]),
        "AP_all_IoU_0.75_maxDets_100": float(coco_eval.stats[2]),
        "AP_small_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[3]),
        "AP_medium_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[4]),
        "AP_large_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[5]),
        "AR_all_IoU_0.50:0.95_maxDets_3": float(coco_eval.stats[6]),
        "AR_all_IoU_0.50:0.95_maxDets_10": float(coco_eval.stats[7]),
        "AR_all_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[8]),
        "AR_small_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[9]),
        "AR_medium_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[10]),
        "AR_large_IoU_0.50:0.95_maxDets_100": float(coco_eval.stats[11])
    }

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        dt_path = sys.argv[1]
        gt = sys.argv[2] if len(sys.argv) > 2 else None
        evaluate(dt_path, gt_path=gt)
    else:
        raise ValueError("Usage: python evaluate.py <detection_results.json> [gt_eval.json]")
