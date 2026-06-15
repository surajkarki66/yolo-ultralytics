# Vitis AI compilation

This directory compiles quantized XIR models (`.xmodel`) for the target DPU on the board. Use it after quantization (see `vitis-ai/quantization/README.md`).

## Structure

- **Architectures/** — DPU architecture JSON files (`arch_B512.json` through `arch_B4096.json`) for different board configurations. Choose the one that matches your target (e.g. `arch_B4096.json` for DPUCZDX8G_ISA1_B4096).
- **model/** — Place quantized `.xmodel` files here before compilation:
  - `DetectionModel_int.xmodel` — YOLOv26 / YOLOv11 detection
  - `OBBModel_int.xmodel` — YOLOv26-OBB / YOLOv11-OBB
- **run_compile.sh** — Example script that invokes `vai_c_xir` to compile the XIR model for a given architecture and output directory.

## Procedure

1. Copy the quantized `.xmodel` from the quantization step into `model/`:
   - Detection: `DetectionModel_int.xmodel`
   - OBB: `OBBModel_int.xmodel`
2. Choose the architecture file from `Architectures/` that matches your target DPU (e.g. B4096, B3136, B2304).
3. Run the compiler, e.g.:

   ```bash
   vai_c_xir -x model/DetectionModel_int.xmodel -a ./Architectures/arch_B4096.json -o zynq_output/yolov26n/ -n yolov26n
   ```

   Or uncomment and run the appropriate line(s) in `run_compile.sh` for your target architecture and model (Detection vs OBB).

4. Use the compiled output in `zynq_output/` (or your chosen `-o` path) for deployment on the device. The evaluation scripts in `vitis-ai/evaluation/` can be used to validate accuracy with the compiled model.
