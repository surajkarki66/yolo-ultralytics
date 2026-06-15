import vart
import xir
import argparse
import pickle
import socket
import struct
import time
import numpy as np

from collections import deque
from typing import List

FRAME_ID_HEADER = struct.Struct("!I")

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    root_subgraph = graph.get_root_subgraph()
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def pack_outputs(frame_id: int, outputs) -> bytes:
    parts = [FRAME_ID_HEADER.pack(frame_id)]
    for arr in outputs:
        parts.append(np.ascontiguousarray(arr, dtype=np.int8).tobytes())
    return b"".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv26 FPGA socket inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--model-path", type=str, required=True, help="Path to YOLOv26 .xmodel")
    parser.add_argument(
        "--task",
        type=str,
        choices=["auto", "detect", "obb"],
        default="auto",
        help="Expected output head type. auto infers from output tensor count.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    graph = xir.Graph.deserialize(args.model_path)
    subgraphs = get_child_subgraph_dpu(graph)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in xmodel")
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")

    input_tensors = dpu_runner.get_input_tensors()
    input_shape = tuple(input_tensors[0].dims)
    input_scale = float(2 ** input_tensors[0].get_attr("fix_point"))
    output_tensors = dpu_runner.get_output_tensors()
    output_shapes = [tuple(t.dims) for t in output_tensors]
    output_scales = [float(2 ** t.get_attr("fix_point")) for t in output_tensors]
    task_mode = args.task
    if task_mode == "auto":
        task_mode = "obb" if len(output_shapes) == 6 else "detect"
    if task_mode == "detect" and len(output_shapes) != 3:
        raise ValueError(f"Detect mode expects 3 output tensors, got {len(output_shapes)}")
    if task_mode == "obb" and len(output_shapes) != 6:
        raise ValueError(f"OBB mode expects 6 output tensors, got {len(output_shapes)}")

    print(f"Server ready: task={task_mode}, input={input_shape}, outputs={output_shapes}")
    print(f"Output scales: {output_scales}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.host, args.port))
    server_socket.listen(1)
    print(f"Waiting for connection on {args.host}:{args.port} ...")

    inference_times = deque(maxlen=100)

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connected to {addr}")
        frame_count = 0
        total_inference_time = 0.0

        try:
            meta = {
                "task": task_mode,
                "input_shape": input_shape,
                "input_scale": input_scale,
                "output_shapes": output_shapes,
                "output_scales": output_scales,
            }
            meta_bytes = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
            client_socket.sendall(struct.pack("!I", len(meta_bytes)) + meta_bytes)

            while True:
                size_data = recv_exact(client_socket, 4)
                if not size_data:
                    break
                payload_size = struct.unpack("!I", size_data)[0]
                payload = recv_exact(client_socket, payload_size)
                if not payload or len(payload) != payload_size:
                    break

                if payload_size < FRAME_ID_HEADER.size:
                    raise ValueError("Bad request payload: too short")
                frame_id = FRAME_ID_HEADER.unpack(payload[: FRAME_ID_HEADER.size])[0]
                input_raw = payload[FRAME_ID_HEADER.size :]
                input_tensor = np.frombuffer(input_raw, dtype=np.int8).reshape(input_shape[1:])
                if input_tensor.shape != tuple(input_shape[1:]):
                    raise ValueError(f"Bad input shape: {input_tensor.shape}, expected {input_shape[1:]}")

                input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
                input_data[0][0, ...] = np.ascontiguousarray(input_tensor, dtype=np.int8)
                output_data = [np.empty(shape, dtype=np.int8, order="C") for shape in output_shapes]

                t0 = time.time()
                jid = dpu_runner.execute_async(input_data, output_data)
                dpu_runner.wait(jid)
                infer_t = time.time() - t0
                inference_times.append(infer_t)
                total_inference_time += infer_t
                frame_count += 1

                response = pack_outputs(frame_id, output_data)
                client_socket.sendall(struct.pack("!I", len(response)) + response)

                if frame_count % 30 == 0:
                    avg_inf = sum(inference_times) / len(inference_times)
                    fpga_fps = 1.0 / avg_inf if avg_inf > 0 else 0.0
                    print(f"Frame {frame_count:4d} | FPGA {fpga_fps:5.1f} fps | latency {avg_inf*1000:5.2f}ms")

        except Exception as exc:
            print(f"Error: {exc}")
        finally:
            client_socket.close()
            if frame_count > 0:
                avg_inf = total_inference_time / frame_count
                fpga_fps = 1.0 / avg_inf if avg_inf > 0 else 0.0
                print(f"Session summary: frames={frame_count}, fpga_fps={fpga_fps:.1f}, latency={avg_inf*1000:.2f}ms")
            print("Client disconnected")


if __name__ == "__main__":
    main()
