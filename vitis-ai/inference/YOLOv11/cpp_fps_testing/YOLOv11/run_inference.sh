#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
#
# Usage: ./run_inference.sh main <MODELFILE>
# Example: ./run_inference.sh main yolov8_int8
#
# Runs from project root. Expects: build_app.sh, build_get_dpu_fps.sh,
# common/, src/, labels.txt. Optional: model/<MODELFILE>.xmodel or <MODELFILE>.xmodel,
# test/ or TEST_IMAGES_DIR for images.

# Always run from script directory so labels.txt, model/, etc. are found
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

MODELFILE=$2

# Path to xmodel: model/<name>.xmodel or <name>.xmodel
get_xmodel_path() {
  if [ -f "model/${MODELFILE}.xmodel" ]; then
    echo "model/${MODELFILE}.xmodel"
  else
    echo "${MODELFILE}.xmodel"
  fi
}

# Images dir for inference
TEST_IMAGES_DIR=${TEST_IMAGES_DIR:-./data}

clean_() {
  echo " "
  echo "cleaning"
  echo " "
  rm -rf test
  rm -f get_dpu_fps cnn_inference
  rm -rf rpt
  rm -f ./*.txt ./*.log 2>/dev/null || true
  mkdir -p rpt
}

# compile_() {
#   echo " "
#   echo "compiling binaries"
#   echo " "
#   bash -x ./build_app.sh
#   bash -x ./build_get_dpu_fps.sh
# }

run_cnn_() {
  echo " "
  echo " run CNN"
  echo " "
  XMODEL=$(get_xmodel_path)
  if [ ! -f "$XMODEL" ]; then
    echo "ERROR: Model not found: $XMODEL"
    return 1
  fi
  if [ ! -f "labels.txt" ]; then
    echo "ERROR: labels.txt not found in $(pwd). Create it or run from project root."
    return 1
  fi
  # C++ app expects: <xmodel> <images_dir/> <labels_file> [output_dir]
  mkdir -p rpt
  OUTPUT_DIR="./rpt/${MODELFILE}_output"
  IMGDIR="${TEST_IMAGES_DIR%/}/"
  ./cnn_inference "$XMODEL" "$IMGDIR" labels.txt "$OUTPUT_DIR" > ./rpt/predictions_${MODELFILE}.log 2>&1
  echo "Predictions log: ./rpt/predictions_${MODELFILE}.log"
  echo "Raw outputs (meta.txt, pred_*_output_*.bin): $OUTPUT_DIR"
  # echo " "
  # echo "FPS performance (1–6 threads)"
  # echo " "
  # ./fps_performance.sh ${MODELFILE}
}

end_() {
  echo " "
  echo "end of inference"
  echo " "
  if [ -d ./test ] && [ -f ./build_testset.sh ]; then
    rm -rf ./test
  fi
}

main() {
  if [ -z "$MODELFILE" ]; then
    echo "Usage: $0 main <MODELFILE>"
    echo "Example: $0 main yolov8_int8"
    exit 1
  fi
  # clean_
  # compile_
  run_cnn_
  # end_
}

"$@"
