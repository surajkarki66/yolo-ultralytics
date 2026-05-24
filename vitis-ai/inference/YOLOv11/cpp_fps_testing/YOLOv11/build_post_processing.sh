#!/bin/sh
# Build post_processing on Vitis AI OS (PetaLinux 2022.2, e.g. xilinx-zcu104).
# Uses same target/prefix logic as build_app.sh. OpenCV only (no glog, no VART).
# Run this on the board or in your PetaLinux/Vitis AI environment.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CXX=${CXX:-g++}

# Same target_info as build_app.sh for install prefix
os=$(lsb_release -a 2>/dev/null | grep "Distributor ID" | sed 's/^.*:\s*//' || true)
os_version=$(lsb_release -a 2>/dev/null | grep "Release" | sed 's/^.*:\s*//' || true)
arch=$(uname -m)
if [ -n "$os" ] && [ -n "$os_version" ]; then
  target_info="${os}.${os_version}.${arch}"
else
  target_info="unknown.${arch}"
fi
install_prefix_default="$HOME/.local/${target_info}"

# OpenCV flags: pkg-config first, then Vitis AI install prefix, then /usr (PetaLinux rootfs)
OPENCV_FLAGS=""
if [ -n "$OPENCV_FLAGS" ]; then
  true
elif command -v pkg-config >/dev/null 2>&1; then
  if pkg-config --exists opencv4 2>/dev/null; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
  elif pkg-config --exists opencv 2>/dev/null; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
  fi
fi

if [ -z "$OPENCV_FLAGS" ]; then
  # PetaLinux / Vitis AI: OpenCV often in rootfs or install prefix (no pkg-config)
  if [ -f "$install_prefix_default.Debug/include/opencv2/opencv.hpp" ] 2>/dev/null || \
     [ -f "$install_prefix_default.Release/include/opencv2/opencv.hpp" ] 2>/dev/null; then
    OPENCV_FLAGS="-I${install_prefix_default}.Debug/include -I${install_prefix_default}.Release/include -L${install_prefix_default}.Debug/lib -L${install_prefix_default}.Release/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
  elif [ -f /usr/include/opencv4/opencv2/opencv.hpp ]; then
    OPENCV_FLAGS="-I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
  elif [ -f /usr/include/opencv2/opencv.hpp ]; then
    OPENCV_FLAGS="-I/usr/include -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
  else
    echo "OpenCV not found. Set OPENCV_FLAGS or install OpenCV on the target."
    echo "Example: OPENCV_FLAGS='-I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui' $0"
    exit 1
  fi
  echo "Using OpenCV flags: $OPENCV_FLAGS"
fi

# Cross-compile / sysroot: same pattern as build_app.sh (Vitis AI SDK)
if echo "$CXX" | grep -q sysroot; then
  $CXX -O2 -std=c++17 -o post_processing \
    "$SCRIPT_DIR/src/post_processing.cc" \
    -I/usr/include/opencv4 \
    -I/install/Debug/include \
    -I/install/Release/include \
    -L/install/Debug/lib \
    -L/install/Release/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
    -lpthread
else
  $CXX -O2 -std=c++17 -o post_processing \
    "$SCRIPT_DIR/src/post_processing.cc" \
    $OPENCV_FLAGS \
    -lpthread
fi

echo "Built: $SCRIPT_DIR/post_processing"
