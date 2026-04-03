"""DPU inference for YOLOv26 Detect (3 outputs) or OBB (6 outputs). Saves int8 tensors to NPZ + optional ``_fps.json``."""

import json
import os
import sys
import time
from glob import glob

import cv2
import numpy as np
import vart
import xir

OBB_NUM_OUTPUTS = 6
OBB_REG_MAX = 16
OBB_STRIDES = [8, 16, 32]


def get_child_subgraph_dpu(graph):
    assert graph is not None
    root = graph.get_root_subgraph()
    assert root is not None
    if root.is_leaf:
        return []
    children = root.toposort_child_subgraph()
    assert children is not None and len(children) > 0
    return [cs for cs in children if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"]


def preprocess_image(image, input_scale, width=416, height=416):
    r = cv2.resize(image, (width, height))
    return ((r.astype(np.float32) / 255.0) * input_scale).astype(np.int8)


def run_fpga_inference(model_path, test_data_path, output_npz_path, img_height=416, img_width=416, obb=False):
    print(f"Loading model from: {model_path}")
    g = xir.Graph.deserialize(model_path)
    subgraphs = get_child_subgraph_dpu(g)
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_tensors = dpu_runner.get_input_tensors()
    input_shape = tuple(input_tensors[0].dims)
    input_scale = 2 ** input_tensors[0].get_attr("fix_point")

    output_tensors = dpu_runner.get_output_tensors()
    output_shapes = [tuple(t.dims) for t in output_tensors]
    output_fixpoints = [t.get_attr("fix_point") for t in output_tensors]

    if obb and len(output_shapes) != OBB_NUM_OUTPUTS:
        print(f"ERROR: OBB expects {OBB_NUM_OUTPUTS} outputs, got {len(output_shapes)}")
        sys.exit(1)

    print(f"Input shape: {input_shape}")
    print(f"Output shapes: {output_shapes}")
    if obb:
        print("Model: OBB (outputs 0-2 detect, 3-5 angle)")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        f for f in glob(os.path.join(test_data_path, "*.*")) if os.path.splitext(f)[1].lower() in exts
    )
    if not image_paths:
        print(f"ERROR: No images in {test_data_path}")
        return

    print(f"Found {len(image_paths)} images, size {img_width}x{img_height}")

    all_names, all_preds = [], []
    t_inf = 0.0
    t0 = time.time()

    for idx, img_path in enumerate(image_paths):
        name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: skip {name}")
            continue

        proc = preprocess_image(image, input_scale, width=img_width, height=img_height)
        input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
        input_data[0][0, ...] = proc.reshape(input_shape[1:])
        output_data = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in output_tensors]

        t1 = time.time()
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        t_inf += time.time() - t1

        all_names.append(name)
        all_preds.append([o.copy() for o in output_data])

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            n = idx + 1
            avg = t_inf / n
            print(f"Processed {n}/{len(image_paths)} | {avg * 1000:.2f} ms/img | {1.0 / avg:.1f} FPS")

    t_all = time.time() - t0
    nimg = len(all_preds)
    if nimg == 0:
        return

    print(f"\nSaving {output_npz_path}...")
    save_dict = {
        "image_names": np.array(all_names),
        "output_shapes": np.array(output_shapes),
        "output_fixpoints": np.array(output_fixpoints),
        "num_outputs": len(output_shapes),
        "model_type": "obb" if obb else "detect",
        "img_height": np.int32(img_height),
        "img_width": np.int32(img_width),
    }
    if obb:
        save_dict["reg_max"] = np.int32(OBB_REG_MAX)
        save_dict["strides"] = np.array(OBB_STRIDES, dtype=np.int32)

    for img_idx, pred_list in enumerate(all_preds):
        for out_idx, pred in enumerate(pred_list):
            save_dict[f"pred_{img_idx}_output_{out_idx}"] = pred

    np.savez_compressed(output_npz_path, **save_dict)
    mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"Done: {nimg} images, {mb:.2f} MB NPZ, total {t_all:.2f}s (DPU {t_inf:.2f}s)")

    fps_path = output_npz_path.replace(".npz", "_fps.json") if output_npz_path.endswith(".npz") else output_npz_path + "_fps.json"
    with open(fps_path, "w") as f:
        json.dump(
            {
                "num_images": nimg,
                "dpu_inference": {
                    "total_time_s": round(t_inf, 4),
                    "avg_per_image_ms": round(t_inf / nimg * 1000, 2),
                    "fps": round(nimg / t_inf, 2),
                },
            },
            f,
            indent=2,
        )
    print(f"FPS stats: {fps_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FPGA/DPU inference to NPZ")
    p.add_argument("model_path")
    p.add_argument("test_data_path")
    p.add_argument("output_npz_path")
    p.add_argument("img_size", type=int)
    p.add_argument("--obb", action="store_true", help="6-output OBB model")
    args = p.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"ERROR: model not found: {args.model_path}")
        sys.exit(1)
    if not os.path.isdir(args.test_data_path):
        print(f"ERROR: folder not found: {args.test_data_path}")
        sys.exit(1)

    run_fpga_inference(
        args.model_path,
        args.test_data_path,
        args.output_npz_path,
        img_height=args.img_size,
        img_width=args.img_size,
        obb=args.obb,
    )
