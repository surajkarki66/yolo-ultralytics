/*

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023
*/

// based on  Vitis AI 3.0 VART "resnet50.cc" demo code

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

string baseImagePath, wordsPath;

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const& path, vector<string>& images) {
  images.clear();
  struct dirent* entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const& path, vector<string>& kinds) {
  kinds.clear();
  ifstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/**
 * @brief Run DPU Task for CNN
 *
 * @return none
 */
void run_CNN(vart::Runner* runner, const std::string& output_dir = "") {

  vector<string> kinds, images;

  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* Load all kinds words.*/
  //LoadWords(wordsPath + "labels.txt", kinds);
  LoadWords(wordsPath, kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: No words exist in file " << wordsPath << endl;
    return;
  }

  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape();
  auto out_dims_1 = outputTensors[1]->get_shape();
  auto out_dims_2 = outputTensors[2]->get_shape();
  auto in_dims = inputTensors[0]->get_shape();

  auto input_scale = get_input_scale(inputTensors[0]);
  int outSize_2 = shapes.outTensorList[2].size;
  int outSize_1 = shapes.outTensorList[1].size;
  int outSize = shapes.outTensorList[0].size;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];

  const bool save_outputs = !output_dir.empty();
  const int num_outputs = static_cast<int>(outputTensors.size());
  if (save_outputs) {
    mkdir(output_dir.c_str(), 0755);
    std::ofstream meta(output_dir + "/meta.txt");
    meta << images.size() << "\n" << num_outputs << "\n";
    for (int i = 0; i < num_outputs; i++) {
      auto dims = outputTensors[i]->get_shape();
      for (size_t d = 0; d < dims.size(); d++) meta << (d ? " " : "") << dims[d];
      meta << "\n";
    }
    for (int i = 0; i < num_outputs; i++)
      meta << (i ? " " : "") << outputTensors[i]->get_attr<int>("fix_point");
    meta << "\n";
    meta.close();
  }

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  int8_t* imageInputs = new int8_t[inSize * batchSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];
  int8_t* FCResult_1 = new int8_t[batchSize * outSize_1];
  int8_t* FCResult_2 = new int8_t[batchSize * outSize_2];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  /*run with batch*/
  for (unsigned int n = 0; n < images.size(); n += batchSize) {
    unsigned int runSize =
        (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
    in_dims[0] = runSize;
    out_dims[0] = batchSize;
    out_dims_1[0] = batchSize;
    out_dims_2[0] = batchSize;
    for (unsigned int i = 0; i < runSize; i++)
    {
      Mat image = imread(baseImagePath + images[n + i]);

      /*image pre-process*/
      Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
      resize(image, image2, Size(inHeight, inWidth), 0, 0, INTER_NEAREST);
      for (int h = 0; h < inHeight; h++) {
        for (int w = 0; w < inWidth; w++) {
          for (int c = 0; c < 3; c++) {
            imageInputs[i*inSize+h*inWidth*3+w*3 + c] = (int8_t)((image2.at<Vec3b>(h, w)[c] / 255.0f) * 2 * input_scale);
          }
        }
      }
    }

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));

    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
	FCResult, batchTensors.back().get()));
    
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
	xir::Tensor::create(outputTensors[1]->get_name(), out_dims_1,
			    xir::DataType{xir::DataType::XINT, 8u})));

    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
	FCResult_1, batchTensors.back().get()));
    
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
	xir::Tensor::create(outputTensors[2]->get_name(), out_dims_2,
			    xir::DataType{xir::DataType::XINT, 8u})));

    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
	FCResult_2, batchTensors.back().get()));

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());
    outputsPtr.push_back(outputs[1].get());
    outputsPtr.push_back(outputs[2].get());
    /*run*/
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

    if (save_outputs) {
      for (unsigned int i = 0; i < runSize; i++) {
        unsigned int img_idx = n + i;
        std::ofstream f0(output_dir + "/pred_" + std::to_string(img_idx) + "_output_0.bin", std::ios::binary);
        f0.write(reinterpret_cast<const char*>(&FCResult[i * outSize]), outSize * sizeof(int8_t));
        std::ofstream f1(output_dir + "/pred_" + std::to_string(img_idx) + "_output_1.bin", std::ios::binary);
        f1.write(reinterpret_cast<const char*>(&FCResult_1[i * outSize_1]), outSize_1 * sizeof(int8_t));
        std::ofstream f2(output_dir + "/pred_" + std::to_string(img_idx) + "_output_2.bin", std::ios::binary);
        f2.write(reinterpret_cast<const char*>(&FCResult_2[i * outSize_2]), outSize_2 * sizeof(int8_t));
      }
    }

    for (unsigned int i = 0; i < runSize; i++) {
      cout << "\nImage : " << images[n + i] << endl;
    }
    inputs.clear();
    outputs.clear();
  }
  delete[] FCResult;
  delete[] FCResult_1;
  delete[] FCResult_2;
  delete[] imageInputs;
}

/**
 * @brief Entry for running CNN
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy CNN on DPU platform.
 *
 */
int main(int argc, char* argv[])
{
  // Check args: 4 required, optional 5th = output dir for raw predictions (same structure as Python, no image_names)
  if (argc < 4 || argc > 5) {
    cout << "Usage: <executable> <xmodel> <test_images_dir> <labels_filename> [output_dir]" << endl;
    return -1;
  }

  baseImagePath = std::string(argv[2]); //path name of the folder with test images
  wordsPath     = std::string(argv[3]); //filename of the labels
  std::string output_dir = (argc >= 5) ? std::string(argv[4]) : "";

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  /*run with batch*/
  run_CNN(runner.get(), output_dir);
  return 0;
}
