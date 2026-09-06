# Vitis AI compilation

Compile quantized `.xmodel` files for your target DPU using `vai_c_xir`.

## Layout

```text
compilation/
  model/                    # input .xmodel files (see model/README.md)
  DPUCZDX8G/               # arch_B512.json … arch_B4096.json
  DPUCZDX8G/Single_Core/    # single-core variants (same B* names)
  run_compile.sh            # commented examples per DPU size
  zynq_output/              # created after compile (gitignored)
```

## Inputs

Copy from `../quantization/quantize_result/` into `model/`:

| Model | File in `model/` |
|-------|-------------------|
| YOLOv8 Detect | `DetectionModel_int.xmodel` |
| YOLOv8-OBB | `OBBModel_int.xmodel` |

## Compile

**Option A — edit `run_compile.sh`**

Uncomment the pair of lines for your DPU (Detect + OBB if needed), then:

```bash
cd vitis-ai/compilation
bash run_compile.sh
```

**Option B — run `vai_c_xir` directly**

Detect (B4096 example):

```bash
cd vitis-ai/compilation
vai_c_xir \
  -x model/DetectionModel_int.xmodel \
  -a DPUCZDX8G/arch_B4096.json \
  -o zynq_output/yolov8n/ \
  -n yolov8n
```

OBB:

```bash
vai_c_xir \
  -x model/OBBModel_int.xmodel \
  -a DPUCZDX8G/arch_B4096.json \
  -o zynq_output/yolov8n_obb/ \
  -n yolov8n_obb
```

Single-core DPU: use `DPUCZDX8G/Single_Core/arch_B4096.json` (or matching `B*`).

## Outputs

Compiled artifacts land under `zynq_output/<name>/`. Copy the deployed `.xmodel` and config pickle to:

- `../evaluation/*/Compiled_Model/` for NPZ-based validation
- `../inference/cpp_fps_testing/*/model/` or `../inference/realtime_inference/` for on-target demos

## Next step

Validate with scripts under [../evaluation/README.md](../evaluation/README.md).
