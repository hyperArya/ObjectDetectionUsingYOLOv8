#use 'pip install ultralytics' to install the required libraries
from ultralytics import YOLO

import cv2

mode = YOLO(r"C:\Users\arshd\Downloads\best.onnx") #download the best.onnx or best.pt and change the file path

result = mode.predict (source="0", show=True)
print(result)
