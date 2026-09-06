import cv2
import socket
import pickle
import struct
import numpy as np
import torch
import time
import os
import argparse
from ultralytics.utils.ops import non_max_suppression


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def decode_bboxes(bboxes, anchors):
    return dist2bbox(bboxes, anchors, xywh=True, dim=1)

def run_model_config(x, pth, name):
    shape = x[0].shape
    with open(os.path.join(pth, f"models/{name}_int8_config_no_srd_reg_nc_dfl.pkl"), 'rb') as f:
        tensor_no, tensor_stride, tensor_reg_max, tensor_nc, layer_dfl = pickle.load(f)

    print(f"🔍 DEBUG run_model_config:")
    print(f"   Input tensor shapes: {[xi.shape for xi in x]}")
    print(f"   tensor_no={tensor_no}, tensor_stride={tensor_stride}")
    print(f"   tensor_reg_max={tensor_reg_max}, tensor_nc={tensor_nc}")
    
    # The view operation should reshape [B, C, H, W] to [B, tensor_no, -1]
    reshaped = []
    for i, xi in enumerate(x):
        print(f"   Tensor {i}: {xi.shape} -> view({shape[0]}, {tensor_no}, -1)")
        reshaped_tensor = xi.view(shape[0], tensor_no, -1)
        print(f"   Result: {reshaped_tensor.shape}")
        reshaped.append(reshaped_tensor)
    
    x_cat = torch.cat(reshaped, 2)
    print(f"   After cat: {x_cat.shape}")
    
    tensor_anchors, tensor_strides = (x.transpose(0, 1) for x in make_anchors(x, tensor_stride, 0.5))
    print(f"   Anchors shape: {tensor_anchors.shape}, Strides shape: {tensor_strides.shape}")
    
    box, cls = x_cat.split((tensor_reg_max * 4, tensor_nc), 1)
    print(f"   Box shape: {box.shape}, Cls shape: {cls.shape}")
    
    dfl_output = layer_dfl(box)
    print(f"   DFL output shape: {dfl_output.shape}")
    print(f"   Anchors unsqueezed: {tensor_anchors.unsqueeze(0).shape}")
    
    dbox = decode_bboxes(layer_dfl(box), tensor_anchors.unsqueeze(0)) * tensor_strides
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y, x

def res_exp(pred, pth, name, iou):
    start = time.time()

    predi = run_model_config(pred, pth, name)
    box_conf = non_max_suppression(prediction=predi, conf_thres=0.6, iou_thres=iou)
    
    if len(box_conf) > 0 and box_conf[0] is not None:
        box_conf[0] = box_conf[0].detach().numpy()
        box_conf[0] = np.delete(box_conf[0], 5, axis=1)
        box_conf[0][box_conf[0] < 0] = 0
        fixed = [[i[0], i[1], i[2]-i[0], i[3]-i[1], i[4]] for i in box_conf[0]]
    else:
        fixed = []
    
    end = time.time() - start
    return fixed, end

class YOLOClient:
    def __init__(self, args):
        self.fpga_host = args.fpga_host
        self.fpga_port = args.fpga_port
        self.config_path = args.config_path
        self.model_name = args.model_name
        self.iou_threshold = args.iou_threshold
        self.img_size = args.img_size
        self.camera_id = args.camera_id
        self.setup_socket()
        
    def setup_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
    def connect(self):
        try:
            self.socket.connect((self.fpga_host, self.fpga_port))
            print(f"✅ Connected to FPGA at {self.fpga_host}:{self.fpga_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to FPGA: {e}")
            return False
        
    def disconnect(self):
        self.socket.close()
        print("🔌 Disconnected from FPGA server")
        
    def send_frame(self, frame):
        try:
            # Resize frame to model input size
            frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
            
            # Encode frame as JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_data = encoded_frame.tobytes()
            
            # Send frame size and data
            frame_size = struct.pack("!I", len(frame_data))
            self.socket.sendall(frame_size + frame_data)
            
            # Receive result size
            result_size_data = self.socket.recv(4)
            if not result_size_data:
                return None
                
            result_size = struct.unpack("!I", result_size_data)[0]
            
            # Receive result data
            result_data = b""
            while len(result_data) < result_size:
                packet = self.socket.recv(min(4096, result_size - len(result_data)))
                if not packet:
                    break
                result_data += packet
            
            if len(result_data) != result_size:
                return None
                
            # The server sends pickled output tensors directly
            output_tensors = pickle.loads(result_data)
            return output_tensors
            
        except Exception as e:
            print(f"❌ Error in send_frame: {e}")
            return None
    
    def post_process_tensors(self, output_tensors):
        """Use your EXACT post-processing pipeline with shape correction"""
        try:
            # Convert numpy tensors to torch tensors
            torch_tensors = []
            
            print(f"🔍 Raw tensor shapes from FPGA: {[t.shape for t in output_tensors]}")
            
            for i, tensor in enumerate(output_tensors):
                # Convert to float32
                torch_tensor = torch.from_numpy(tensor.astype(np.float32))
                
                print(f"   Tensor {i} before transpose: {torch_tensor.shape}")
                
                if len(torch_tensor.shape) == 4:
                    _, dim1, dim2, _ = torch_tensor.shape
                
                    is_nhwc = (dim1 > 10 and dim2 > 10 and abs(dim1 - dim2) <= 1)
                    
                    if is_nhwc:
                        print(f"   🔄 Detected NHWC format [B,H,W,C], transposing to NCHW [B,C,H,W]...")
                        # Permute from [B, H, W, C] to [B, C, H, W]
                        torch_tensor = torch_tensor.permute(0, 3, 1, 2)
                        print(f"   ✅ After transpose: {torch_tensor.shape}")
                    else:
                        print(f"   ℹ️  Detected NCHW format [B,C,H,W], no transpose needed")
                
                torch_tensors.append(torch_tensor)
            
            print(f"📊 Final tensor shapes for processing: {[t.shape for t in torch_tensors]}")
            
            # Verify shapes are correct before processing
            expected_shapes = [(1, 65, 52, 52), (1, 65, 26, 26), (1, 65, 13, 13)]
            actual_shapes = [tuple(t.shape) for t in torch_tensors]
            
            if actual_shapes != expected_shapes:
                print(f"⚠️  WARNING: Unexpected tensor shapes!")
                print(f"   Expected: {expected_shapes}")
                print(f"   Got: {actual_shapes}")
            
            # Use your EXACT res_exp function
            detections, processing_time = res_exp(torch_tensors, self.config_path, self.model_name, self.iou_threshold)
            
            print(f"✅ Found {len(detections)} detections in {processing_time:.3f}s")
            return detections
            
        except Exception as e:
            print(f"❌ Error in post_process_tensors: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detections with 'person' label instead of confidence number"""
        if detections:
            H, W = frame.shape[:2]
            scale_x = W / self.img_size
            scale_y = H / self.img_size
            
            for box in detections:
                if len(box) >= 5:  # x, y, w, h, conf
                    x, y, w, h, conf = box
                    
                    # Scale coordinates exactly like your code
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Display "person" instead of confidence number
                    label = f"person {conf:.2f}"  # You can remove confidence if you want just "person"
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        frame, 
                        (x, y - text_height - 5), 
                        (x + text_width, y), 
                        (0, 255, 0), 
                        -1  # Filled rectangle
                    )
                    
                    # Draw "person" text
                    cv2.putText(
                        frame, 
                        label, 
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0),  # Black text for better visibility
                        1
                    )
        
        return frame
    
    def run(self):
        if not self.connect():
            return
        
        # Open webcam
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera (ID: {self.camera_id})")
            self.disconnect()
            return
        
        print("🎥 Camera opened successfully")
        print("🚀 Starting real-time detection. Press 'q' to quit")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Send frame to FPGA and get results
                output_tensors = self.send_frame(frame)
                
                if output_tensors and len(output_tensors) == 3:
                    # Post-process results using your EXACT pipeline
                    detections = self.post_process_tensors(output_tensors)
                    
                    # Draw detections using your EXACT visualization
                    frame_with_detections = self.draw_detections(frame.copy(), detections)
                    
                    # Calculate and display FPS
                    frame_count += 1
                    current_time = time.time()
                    fps = frame_count / (current_time - start_time)
                    
                    # Display information
                    cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_with_detections, f"Detections: {len(detections)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("YOLO Real-time Detection", frame_with_detections)
                else:
                    cv2.imshow("YOLO Real-time Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.disconnect()
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"\n📊 Session Summary:")
            print(f"   Frames processed: {frame_count}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average FPS: {avg_fps:.1f}")

def main():
    parser = argparse.ArgumentParser(description='YOLO Client with Exact Post-Processing')
    
    # Required arguments - matching your original code
    parser.add_argument('--fpga-host', type=str, required=True,
                       help='IP address of the FPGA server')
    parser.add_argument('--config-path', type=str, required=True,
                       help='Path to directory containing models folder')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name for config file (e.g., yolov8n)')
    
    # Optional arguments with defaults
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                       help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=416,
                       help='Input image size')
    parser.add_argument('--fpga-port', type=int, default=8888,
                       help='FPGA server port')
    parser.add_argument('--camera-id', type=int, default=2,
                       help='Camera ID')
    
    args = parser.parse_args()
    
    # Validate that the config file exists
    config_file = os.path.join(args.config_path, f"models/{args.model_name}_int8_config_no_srd_reg_nc_dfl.pkl")
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file not found at {config_file}")
        print("💡 Make sure your pickle file is in the models/ subdirectory")
        return
    
    print("🔧 Configuration:")
    print(f"   FPGA Server: {args.fpga_host}:{args.fpga_port}")
    print(f"   Config Path: {args.config_path}")
    print(f"   Model Name: {args.model_name}")
    print(f"   IoU Threshold: {args.iou_threshold}")
    print(f"   Image Size: {args.img_size}")
    print(f"   Camera ID: {args.camera_id}")
    print()
    
    client = YOLOClient(args)
    client.run()

if __name__ == "__main__":
    main()
