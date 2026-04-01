import sys
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(dt_path, save_path="metrics.json"):
    # Load detection results
    if isinstance(dt_path, str):
        dt = json.load(open(dt_path))
    else:
        dt = dt_path

    assert isinstance(dt, dict), 'Detection result format is not correct.'

    # Load COCO ground truth
    coco_gt = COCO('gt_eval.json')
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
            dt_coco.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": box[:4],
                "score": box[-1]
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
    if len(sys.argv) == 2:
        dt_path = sys.argv[1]
        evaluate(dt_path)
    else:
        raise ValueError("Usage: python evaluate.py <detection_results.json>")
