# Vitis AI evaluation

Post-processing and metrics for quantized YOLOv8 **Detect** and **OBB** models.

Two deployment paths:

| Path | When to use | Raw outputs |
|------|-------------|-------------|
| **Quantized_Model** | ONNX (or PC-side quantized model) | Run ONNX → tensors |
| **Compiled_Model** | DPU / FPGA after compilation | `fpga_inference.py` → `.npz` |

Both paths need the **config pickle** from quantization:

`quantize_result/{checkpoint_stem}_config_no_srd_reg_nc_dfl.pkl`  
(e.g. `best_config_no_srd_reg_nc_dfl.pkl` for `best.pt`)

## Detect pipeline

### Quantized (ONNX)

```text
post_processing_onnx.py  →  output_boxes_*.txt
coco_prep.py             →  coco-preped-*.json
evaluate.py              →  COCO metrics (pycocotools)
```

Example:

```bash
cd vitis-ai/evaluation/Detect/Quantized_Model
python post_processing_onnx.py \
  --onnx path/to/model.onnx \
  --pkl path/to/best_config_no_srd_reg_nc_dfl.pkl \
  --images-dir /path/to/images \
  --iou 0.45 --img-height 416 --img-width 416
python coco_prep.py --input output_boxes_<stem>.txt --gt gt_eval.json
python evaluate.py coco-preped-<stem>.json gt_eval.json
```

Optional: `visualize_output.py` for box overlays.

### Compiled (FPGA NPZ)

```text
fpga_inference.py   →  predictions.npz
post_processing.py  →  output_boxes_*.txt
coco_prep.py        →  coco-preped-*.json
evaluate.py         →  COCO metrics
```

Example:

```bash
cd vitis-ai/evaluation/Detect/Compiled_Model
python fpga_inference.py <compiled.xmodel> <image_list_or_dir> predictions.npz 416
python post_processing.py --npz predictions.npz --pkl path/to/best_config_no_srd_reg_nc_dfl.pkl --iou 0.45
python coco_prep.py --input output_boxes_predictions.txt --gt gt_eval.json
python evaluate.py coco-preped-output_boxes_predictions.json gt_eval.json
```

`coco_mapping.py` maps YOLO class indices to COCO category IDs using `gt_eval.json`.

## OBB pipeline

### Quantized (ONNX)

```bash
cd vitis-ai/evaluation/OBB/Quantized_Model
python post_processing.py --onnx ... --config ... --images-dir ...
python evaluate.py --data-dir /path/to/dataset --split val ...
```

Metrics are computed in-script (no pycocotools). Optional annotated image export via `--save-annotated`.

### Compiled (FPGA NPZ)

```bash
cd vitis-ai/evaluation/OBB/Compiled_Model
python fpga_inference.py <compiled.xmodel> <test_data> output.npz 416 --obb
python evaluate_obb.py --data-dir /path/to/dataset --split val --npz output.npz --config path/to/config.pkl
```

Visualization only:

```bash
python inference_obb.py --npz output.npz --config config.pkl --images-dir ... --output-dir runs/inference_obb
```

Shared helpers: `utils/util.py`, `utils/obb_utils.py` (evaluation-only; training code removed from Compiled `util.py`).

## Folder reference

```text
evaluation/
  Detect/
    Quantized_Model/
      post_processing_onnx.py
      coco_prep.py
      evaluate.py
      coco_mapping.py
      visualize_output.py
      utils.py
    Compiled_Model/
      fpga_inference.py
      post_processing.py
      coco_prep.py
      evaluate.py
      coco_mapping.py
      visualize_output.py
  OBB/
    Quantized_Model/
      post_processing.py
      evaluate.py
      utils/
    Compiled_Model/
      fpga_inference.py
      post_processing_obb.py
      evaluate_obb.py
      inference_obb.py
      utils/
```

## Dependencies

- **Detect + COCO:** `pycocotools` for `evaluate.py`
- **FPGA scripts:** `vart`, `xir` (Vitis AI runtime on board or target filesystem)
- **ONNX path:** `onnxruntime`
- **OBB:** `torch`, `opencv-python`, `pyyaml`

Install per-script needs in your Vitis or board environment.
