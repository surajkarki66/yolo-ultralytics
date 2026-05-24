# Model inspection (Vitis AI NNDCT)

Run the **NNDCT `Inspector`** on a PyTorch checkpoint **before** quantization. It checks whether the graph and operators are compatible with a chosen **DPU target** and writes visualization artifacts (PNG by default) so you can spot unsupported layers early.

Applies to **YOLOv26**, **YOLOv11**, and their OBB variants.

## Prerequisites

- **Vitis AI** environment with PyTorch and `pytorch_nndct` (same Docker / conda stack as quantization), e.g. `conda activate vitis-ai-pytorch`.
- Patch Ultralytics in the Vitis environment first (see `../quantization/README.md`).
- **Checkpoint format**: `inspection.py` loads `torch.load(...)` and expects a dict with a **`"model"`** key (Ultralytics training output, same as `../quantization/vai_q_yolo.py`).
- Activations are already DPU-friendly when trained with root `custom_layers/` + `patch_ultralytics.py` (HardSwish via patched modules).

## Script: `inspection.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | Path to the `.pt` checkpoint (e.g. `best.pt`) |
| `--target` | `DPUCZDX8G_ISA1_B4096` | DPU target (must match quantization and compilation) |
| `--img_height` | `416` | Input height (match training / quantization) |
| `--img_width` | `416` | Input width (match training / quantization) |
| `--output_dir` | `inspect` | Folder for inspector PNG output |

The script casts weights to **float32**, sets **eval** mode, builds a random tensor of shape `[1, 3, H, W]`, and calls:

`Inspector(target).inspect(model, (dummy_input,), device=..., output_dir=..., image_format="png")`

**Output**: files under **`inspect/`** (or `--output_dir`) in the current working directory. Treat this folder as generated output.

Example:

```bash
cd vitis-ai/inspection
conda activate vitis-ai-pytorch
python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
```

Align `--img_height` / `--img_width` and `--target` with `../quantization/vai_q_yolo.py` and `../compilation/Architectures/arch_B*.json`.

## Reference commands: `inspect.sh`

`inspect.sh` is a commented cheat sheet of example invocations for:

- **YOLOv26 / YOLOv11** detection (`best.pt`) across DPU sizes **B4096** down to **B512**
- **YOLOv26-OBB / YOLOv11-OBB** with the same targets

Uncomment or copy the line you need.

## Workflow position

1. Train in the main repo (`python main.py train`) with patched `custom_layers/`.
2. Copy `best.pt` into this folder (or pass an absolute path).
3. Patch Ultralytics in `../quantization/` if not already done.
4. Run **`inspection.py`** here to validate DPU feasibility for your `--target`.
5. Run quantization in `../quantization/`, then compilation in `../compilation/model/`.

## Layout

| Path | Role |
|------|------|
| `inspection.py` | NNDCT inspector entry point |
| `inspect.sh` | Example commands per model type and DPU |
| `inspect/` | Default output directory for inspector PNGs (generated) |
