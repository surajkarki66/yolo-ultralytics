## B4096

## YOLOv26 / YOLOv11 Detect — NMS (one2many, default)
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B4096_nms

## YOLOv26 / YOLOv11 Detect — end2end (NMS-free top-k)
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B4096_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS (one2many, default)
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B4096_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end (NMS-free top-k)
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B4096_end2end


## B3136

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B3136_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B3136_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B3136_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B3136_end2end


## B2304

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B2304_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B2304_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B2304_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B2304_end2end


## B1600

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1600_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1600_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1600_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1600_end2end


## B1152

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1152_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1152_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1152_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1152_end2end


## B1024

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1024_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B1024_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1024_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B1024_end2end


## B800

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B800_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B800_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B800_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B800_end2end


## B512

## YOLOv26 / YOLOv11 Detect — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B512_nms

## YOLOv26 / YOLOv11 Detect — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_416_B512_end2end

## YOLOv26-OBB / YOLOv11-OBB — NMS
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B512_nms

## YOLOv26-OBB / YOLOv11-OBB — end2end
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib --end2end
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy --end2end
#sleep 20
#mv quantize_result quantize_result_yolov26n_obb_416_B512_end2end
