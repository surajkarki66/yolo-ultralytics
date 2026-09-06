#!/bin/bash
# Copy best.pt into this directory or pass a path to your trained checkpoint.
# Run after patching Ultralytics in vitis-ai/quantization (see README.md).

## B4096

## YOLOv8
python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512

## YOLOv8-OBB
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800
#python inspection.py --model_path best.pt --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512
