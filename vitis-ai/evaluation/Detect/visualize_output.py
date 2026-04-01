import cv2
import os
import sys

def visualize_boxes(pred_file, image_folder="test_data", save_folder="output_visualized", img_height=416, img_width=416):
    os.makedirs(save_folder, exist_ok=True)

    with open(pred_file, "r") as f:
        lines = f.readlines()

    image_name = None
    boxes = []

    for line in lines:
        line = line.strip()
        if line.startswith("Image Prediction:"):
            if image_name and boxes:
                img_path = os.path.join(image_folder, image_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Image not found: {img_path}")
                else:
                    H, W = img.shape[:2]
                    scale_x = W / img_width
                    scale_y = H / img_height
                    for box in boxes:
                        x, y, w, h, conf = map(float, box)
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(img, f"{conf:.2f}", (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.imwrite(os.path.join(save_folder, image_name), img)
                boxes = []

            image_name = line.split(":")[1].strip()
        elif line and not line.startswith("Boxes:"):
            boxes.append(line.split())

    # Draw the last image
    if image_name and boxes:
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        if img is not None:
            H, W = img.shape[:2]
            scale_x = W / img_width
            scale_y = H / img_height
            for box in boxes:
                x, y, w, h, conf = map(float, box)
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(os.path.join(save_folder, image_name), img)

    print(f"All images saved in '{save_folder}'")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_output.py <pred_file> <img_height> <img_width>")
        sys.exit(1)

    pred_file = sys.argv[1]
    image_folder = "test_data"
    save_folder = "output_visualized"
    img_height = int(sys.argv[2])
    img_width  = int(sys.argv[3]) 

    visualize_boxes(pred_file, image_folder, save_folder, img_height, img_width)
