import cv2
import glob
import os

# Folder containing images
image_folder = "./out"

# Output video file
video_name = "output.mp4"

# Frames per second
fps = 20

# Get sorted list of images
images = sorted(glob.glob(os.path.join(image_folder, "frame_*.jpg")))

# Read first image to get size
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Write frames
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release video
video.release()

print(f"Video saved as {video_name}")
