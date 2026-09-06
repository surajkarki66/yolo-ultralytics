## B4096

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B4096

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B4096 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B4096


## B3136

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B3136

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B3136 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B3136


## B2304

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B2304

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B2304 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B2304


## B1600

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B1600

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1600 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B1600


## B1152

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B1152

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1152 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B1152


## B1024

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B1024

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B1024 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B1024


## B800

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B800

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B800 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B800


## B512

## YOLOv8
python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib
sleep 20
python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy
sleep 20
mv quantize_result quantize_result_yolov8n_416_B512

## YOLOv8-OBB
#python vai_q_yolo.py --model_path "best.pt" --batch_size 16 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode calib
#sleep 20
#python vai_q_yolo.py --model_path "best.pt" --batch_size 1 --img_height 416 --img_width 416 --target DPUCZDX8G_ISA1_B512 --quant_mode test --deploy
#sleep 20
#mv quantize_result quantize_result_yolov8n_obb_416_B512
