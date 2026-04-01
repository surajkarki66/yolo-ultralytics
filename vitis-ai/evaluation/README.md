# Vitis AI evaluation (YOLOv26)

Host-side and DPU-side scripts to run **Detect** or **OBB** models after training, ONNX export, quantization, and compilation. The typical flow is:

1. **ONNX on GPU/CPU** — validate decoding and mAP with `eval_onnx.py`.
2. **DPU / FPGA** — run raw inference to `predictions.npz` with `fpga_inference.py` or `fpga_inference_obb.py`.
3. **Decode NPZ on host** — same post-processing as ONNX using `eval_predictions_npz.py`.

Shared helpers live in `utils.py` (metrics, NMS, YAML, IoU).

## Layout

| File | Role |
|------|------|
| `eval_onnx.py` | ONNX Runtime: YOLOv26 Detect or OBB, optional mAP, visualizations. |
| `eval_predictions_npz.py` | Decode int8 DPU outputs from `.npz`, optional mAP, matches FPGA preprocessing. |
| `fpga_inference.py` | Detect-only DPU runner → compressed `.npz` + `_fps.json`. |
| `fpga_inference_obb.py` | Detect or OBB (`--obb`, six outputs) DPU runner → `.npz`. |
| `utils.py` | `DetMetrics`, `OBBMetrics`, NMS, YAML helpers, shared decode utilities. |
| `model/remove_dfl.py` | Optional pickle cleaner for quantization configs (removes `DFL` objects). |

## Dependencies

- **Host eval (ONNX / NPZ):** `onnxruntime`, `opencv-python`, `numpy`, `torch`, and packages already required by `utils.py` (e.g. Polars if you use export helpers).
- **DPU scripts:** AMD Vitis AI runtime (`xir`, `vart`) and OpenCV on the **target** or a machine with the DPU stack installed.

Run evaluation scripts from this directory so imports resolve:

```bash
cd vitis-ai/evaluation
```

## 1. ONNX evaluation

Uses the validation (or test) image list from your dataset YAML (`val` / `test`).

**OBB (default task):**

```bash
python eval_onnx.py --task obb --model /path/to/model.onnx --data /path/to/data.yaml \
  --imgsz 416 --eval-map --save-dir runs/obb_onnx_eval
```

**Detect:**

```bash
python eval_onnx.py --task detect --model /path/to/model.onnx --data /path/to/data.yaml \
  --imgsz 416 --eval-map --save-dir runs/detect_onnx_eval
```

**Quantization metadata:** if you have `*_config_no_srd_reg_nc_dfl.pkl` from the quantizer, pass:

```bash
--quant-meta /path/to/your_config_no_srd_reg_nc_dfl.pkl
```

so strides, `reg_max`, and class count match the exported graph.

## 2. DPU inference → NPZ

**Detect (three outputs), square input size (e.g. 416):**

```bash
python fpga_inference.py /path/to/model.xmodel /path/to/images /path/to/predictions.npz 416
```

**Detect or OBB with the unified script:**

```bash
python fpga_inference_obb.py /path/to/model.xmodel /path/to/images /path/to/predictions.npz 416
python fpga_inference_obb.py ... 416 --obb   # YOLOv26-OBB: six outputs
```

Outputs include per-image int8 tensors and fix-point scales; see the module docstrings in each file for key names.

## 3. Evaluate NPZ on the host

Point to the same dataset YAML used for labels. Match preprocessing to the FPGA (`resize` vs `letterbox`).

**OBB:**

```bash
python eval_predictions_npz.py --predictions-npz /path/to/predictions.npz --data /path/to/data.yaml \
  --task obb --imgsz 416 --preprocess resize --eval-map
```

**Detect:**

```bash
python eval_predictions_npz.py --predictions-npz /path/to/predictions.npz --data /path/to/data.yaml \
  --task detect --imgsz 416 --preprocess resize --eval-map
```

If image basenames in the NPZ do not sit under the default `val` folder from the YAML, set:

```bash
--images-root /path/to/dataset/images
```

Optional `--quant-meta` behaves like ONNX eval. OBB NPZ files may already contain `reg_max` / `strides`; the script uses them when `--quant-meta` is omitted.

## 4. Pickle helper (`model/remove_dfl.py`)

Only needed if your workflow produces a pickle that embeds `DFL` modules and downstream tools fail to load it. **Back up** `config_yolov26.pkl`, then:

```bash
cd vitis-ai/evaluation/model
python remove_dfl.py
```

Paths are configurable at the top of the script.

## See also

- Quantization: `../quantization/README.md`
- Compilation: `../compilation/README.md`
- Root project README for the full training → quantize → compile pipeline
