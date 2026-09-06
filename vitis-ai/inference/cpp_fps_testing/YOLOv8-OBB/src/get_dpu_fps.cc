/*
 * Optimized DPU Benchmark — YOLOv8-OBB (6 outputs)
 * outputs 0,1,2 = detection heads
 * outputs 3,4,5 = angle heads
 * - Accurate latency (per inference)
 * - Accurate FPS
 * - No tensor/buffer creation inside loop
 */

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <assert.h>

#include "common.h"

using namespace std;
using namespace std::chrono;

GraphInfo shapes;

int num_threads       = 0;
int num_of_images     = 0;
int num_images_x_thread = 0;

// global timing accumulators
mutex mtx;
double total_dpu_time = 0.0;

// YOLOv8-OBB: 6 outputs — detect heads (0,1,2) + angle heads (3,4,5)
void runDPU(vart::Runner* runner,
            int8_t *imageInputs,
            int8_t *FCResult,   // detect head 0
            int8_t *FCResult1,  // detect head 1
            int8_t *FCResult2,  // detect head 2
            int8_t *FCResult3,  // angle head 0
            int8_t *FCResult4,  // angle head 1
            int8_t *FCResult5)  // angle head 2
{
  auto inputTensors  = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  auto in_dims   = inputTensors[0]->get_shape();
  auto out_dims  = outputTensors[0]->get_shape();
  auto out_dims1 = outputTensors[1]->get_shape();
  auto out_dims2 = outputTensors[2]->get_shape();
  auto out_dims3 = outputTensors[3]->get_shape();
  auto out_dims4 = outputTensors[4]->get_shape();
  auto out_dims5 = outputTensors[5]->get_shape();

  int batchSize = in_dims[0];

  int inSize   = shapes.inTensorList[0].size;
  int outSize  = shapes.outTensorList[0].size;
  int outSize1 = shapes.outTensorList[1].size;
  int outSize2 = shapes.outTensorList[2].size;
  int outSize3 = shapes.outTensorList[3].size;
  int outSize4 = shapes.outTensorList[4].size;
  int outSize5 = shapes.outTensorList[5].size;

  // Create tensors ONCE (outside the inference loop)
  auto inTensor   = xir::Tensor::create(inputTensors[0]->get_name(),  in_dims,   xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor  = xir::Tensor::create(outputTensors[0]->get_name(), out_dims,  xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor1 = xir::Tensor::create(outputTensors[1]->get_name(), out_dims1, xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor2 = xir::Tensor::create(outputTensors[2]->get_name(), out_dims2, xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor3 = xir::Tensor::create(outputTensors[3]->get_name(), out_dims3, xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor4 = xir::Tensor::create(outputTensors[4]->get_name(), out_dims4, xir::DataType{xir::DataType::XINT, 8u});
  auto outTensor5 = xir::Tensor::create(outputTensors[5]->get_name(), out_dims5, xir::DataType{xir::DataType::XINT, 8u});

  // Create buffers ONCE (outside the inference loop)
  auto inputBuffer   = std::make_unique<CpuFlatTensorBuffer>(imageInputs, inTensor.get());
  auto outputBuffer  = std::make_unique<CpuFlatTensorBuffer>(FCResult,    outTensor.get());
  auto outputBuffer1 = std::make_unique<CpuFlatTensorBuffer>(FCResult1,   outTensor1.get());
  auto outputBuffer2 = std::make_unique<CpuFlatTensorBuffer>(FCResult2,   outTensor2.get());
  auto outputBuffer3 = std::make_unique<CpuFlatTensorBuffer>(FCResult3,   outTensor3.get());
  auto outputBuffer4 = std::make_unique<CpuFlatTensorBuffer>(FCResult4,   outTensor4.get());
  auto outputBuffer5 = std::make_unique<CpuFlatTensorBuffer>(FCResult5,   outTensor5.get());

  std::vector<vart::TensorBuffer*> inputsPtr  = {inputBuffer.get()};
  std::vector<vart::TensorBuffer*> outputsPtr = {
      outputBuffer.get(),
      outputBuffer1.get(),
      outputBuffer2.get(),
      outputBuffer3.get(),  // angle head 0
      outputBuffer4.get(),  // angle head 1
      outputBuffer5.get()   // angle head 2
  };

  double thread_time = 0.0;

  for (int n = 0; n < num_images_x_thread; n += batchSize)
  {
    auto t1 = high_resolution_clock::now();

    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);

    auto t2 = high_resolution_clock::now();

    duration<double, micro> dpu_time = t2 - t1;
    thread_time += dpu_time.count();
  }

  // Accumulate thread results safely
  lock_guard<mutex> lock(mtx);
  total_dpu_time += thread_time;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    cout << "Usage: ./app model.xmodel num_threads num_images\n";
    return -1;
  }

  num_threads   = atoi(argv[2]);
  num_of_images = atoi(argv[3]);

  auto graph    = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());

  auto runner  = vart::Runner::create_runner(subgraph[0], "run");
  auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner3 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner4 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner5 = vart::Runner::create_runner(subgraph[0], "run");

  auto inputTensors  = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  int inputCnt  = inputTensors.size();
  int outputCnt = outputTensors.size();

  // Sanity-check: must be 6 outputs for YOLOv8-OBB
  if (outputCnt != 6) {
    cout << "ERROR: Expected 6 outputs for YOLOv8-OBB (got " << outputCnt << ")\n";
    return -1;
  }

  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];

  shapes.inTensorList  = inshapes;
  shapes.outTensorList = outshapes;

  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  int inSize   = shapes.inTensorList[0].size;
  int outSize  = shapes.outTensorList[0].size;
  int outSize1 = shapes.outTensorList[1].size;
  int outSize2 = shapes.outTensorList[2].size;
  int outSize3 = shapes.outTensorList[3].size;
  int outSize4 = shapes.outTensorList[4].size;
  int outSize5 = shapes.outTensorList[5].size;

  int batchSize = inputTensors[0]->get_shape()[0];

  num_images_x_thread = (num_of_images / num_threads / batchSize) * batchSize;
  num_of_images       = num_images_x_thread * num_threads;

  // Allocate memory for inputs and all 6 outputs
  int8_t *imageInputs = new int8_t[num_of_images * inSize];
  int8_t *FCResult    = new int8_t[num_of_images * outSize];
  int8_t *FCResult_1  = new int8_t[num_of_images * outSize1];
  int8_t *FCResult_2  = new int8_t[num_of_images * outSize2];
  int8_t *FCResult_3  = new int8_t[num_of_images * outSize3];  // angle head 0
  int8_t *FCResult_4  = new int8_t[num_of_images * outSize4];  // angle head 1
  int8_t *FCResult_5  = new int8_t[num_of_images * outSize5];  // angle head 2

  thread workers[num_threads];

  auto t_start = high_resolution_clock::now();

  for (int i = 0; i < num_threads; i++)
  {
    vart::Runner* r =
      (i==0)?runner.get():
      (i==1)?runner1.get():
      (i==2)?runner2.get():
      (i==3)?runner3.get():
      (i==4)?runner4.get():
             runner5.get();

    int offset = i * num_images_x_thread;

    workers[i] = thread(runDPU,
                        r,
                        imageInputs + offset * inSize,
                        FCResult    + offset * outSize,
                        FCResult_1  + offset * outSize1,
                        FCResult_2  + offset * outSize2,
                        FCResult_3  + offset * outSize3,
                        FCResult_4  + offset * outSize4,
                        FCResult_5  + offset * outSize5);
  }

  for (auto &t : workers) t.join();

  auto t_end = high_resolution_clock::now();

  duration<double> total_time = t_end - t_start;

  double fps             = num_of_images / total_time.count();
  double avg_latency_us  = total_dpu_time / num_of_images;

  cout << "\n===== FINAL RESULTS (YOLOv8-OBB) =====\n";
  cout << "Total Images        : " << num_of_images << endl;
  cout << "Total Time (sec)    : " << total_time.count() << endl;
  cout << "FPS                 : " << fps << endl;
  cout << "Avg Latency (us)    : " << avg_latency_us << endl;
  cout << "Avg Latency (ms)    : " << avg_latency_us / 1000.0 << endl;
  cout << "Output heads        : detect[0,1,2]  angle[3,4,5]\n";
  cout << "=======================================\n";

  delete[] imageInputs;
  delete[] FCResult;
  delete[] FCResult_1;
  delete[] FCResult_2;
  delete[] FCResult_3;
  delete[] FCResult_4;
  delete[] FCResult_5;

  return 0;
}
