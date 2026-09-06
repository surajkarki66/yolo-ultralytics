import cv2
import numpy as np
import vart
import xir
import socket
import struct
import pickle
import time
from typing import List
from collections import deque

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert root_subgraph is not None, "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def preprocess_frame(frame, input_scale, width=416, height=416):
    # Resize to YOLO input shape (416x416)
    frame_resized = cv2.resize(frame, (width, height))
    
    # Normalize pixel values to [0, 1] and scale to int8
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_scaled = (frame_normalized * input_scale).astype(np.int8)
    
    return frame_scaled

# Setup model
model_path = "yv8/yolov8n.xmodel"
g = xir.Graph.deserialize(model_path)
subgraphs = get_child_subgraph_dpu(g)
dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

# Get input tensor info
inputTensors = dpu_runner.get_input_tensors()
input_shape = tuple(inputTensors[0].dims)
input_fixpos = inputTensors[0].get_attr("fix_point")
input_scale = 2**input_fixpos

# Get output tensor info
outputTensors = dpu_runner.get_output_tensors()
output_shapes = [tuple(tensor.dims) for tensor in outputTensors]

print(f"Server ready: Input shape {input_shape}, Output shapes {output_shapes}")

# Socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8888))
server_socket.listen(1)
print("Waiting for connection...")

# FPS tracking
frame_count = 0
inference_times = deque(maxlen=100)
total_inference_time = 0

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connected to {addr}")
    print(f"Starting FPS measurement...\n")
    
    # Reset counters
    frame_count = 0
    inference_times.clear()
    total_inference_time = 0
    session_start = time.time()
    
    try:
        while True:
            # Receive frame size (4 bytes)
            size_data = client_socket.recv(4)
            if not size_data:
                break
                
            frame_size = struct.unpack("!I", size_data)[0]
            
            # Receive frame data
            frame_data = b""
            while len(frame_data) < frame_size:
                chunk = client_socket.recv(min(4096, frame_size - len(frame_data)))
                if not chunk:
                    break
                frame_data += chunk
            
            if len(frame_data) != frame_size:
                print("Incomplete frame received")
                break
            
            # Decode frame
            frame_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Failed to decode frame")
                client_socket.sendall(struct.pack("!I", 0))
                continue
            
            # Preprocess
            processed_frame = preprocess_frame(frame, input_scale)
            
            # Prepare input data
            input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
            input_data[0][0, ...] = processed_frame.reshape(input_shape[1:])
            
            # Prepare output data
            output_data = []
            for outputTensor in outputTensors:
                output_shape = tuple(outputTensor.dims)
                output_data.append(np.empty(output_shape, dtype=np.int8, order="C"))
            
            # ⏱️ PURE FPGA INFERENCE TIME
            inference_start = time.time()
            job_id = dpu_runner.execute_async(input_data, output_data)
            dpu_runner.wait(job_id)
            inference_time = time.time() - inference_start
            
            # Track inference time
            inference_times.append(inference_time)
            total_inference_time += inference_time
            frame_count += 1
            
            # Pack all three tensors into a list with pickle
            result_data = pickle.dumps(output_data)
            
            # Send the size of pickled data first, then the data
            client_socket.sendall(struct.pack("!I", len(result_data)))
            client_socket.sendall(result_data)
            
            # Display FPS every 30 frames
            if frame_count % 30 == 0:
                # Pure FPGA metrics
                avg_inference_time = sum(inference_times) / len(inference_times)
                fpga_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                
                # Overall metrics
                session_time = time.time() - session_start
                overall_fps = frame_count / session_time if session_time > 0 else 0
                
                print(f"Frame {frame_count:4d} | 🚀 FPGA: {fpga_fps:5.1f} fps ({avg_inference_time*1000:5.2f}ms) | Overall: {overall_fps:5.1f} fps")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        
        # Final statistics
        if frame_count > 0:
            session_time = time.time() - session_start
            avg_inference_time = total_inference_time / frame_count
            fpga_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
            overall_fps = frame_count / session_time if session_time > 0 else 0
            
            print(f"\n{'='*70}")
            print(f"SESSION SUMMARY")
            print(f"{'='*70}")
            print(f"Total frames: {frame_count}")
            print(f"Session time: {session_time:.2f}s")
            print(f"Total inference time: {total_inference_time:.2f}s")
            print()
            print(f"🚀 PURE FPGA PERFORMANCE:")
            print(f"   Avg inference: {avg_inference_time*1000:.2f}ms")
            print(f"   FPGA FPS: {fpga_fps:.1f}")
            print()
            print(f"📈 OVERALL:")
            print(f"   Overall FPS: {overall_fps:.1f}")
            print(f"   Inference: {(total_inference_time/session_time)*100:.1f}%")
            print(f"{'='*70}\n")
        
        print("Client disconnected")
