from ultralytics import YOLO

import cv2

mode = YOLO(r"C:\Users\arshd\Downloads\best.onnx")

result = mode.predict (source="0", show=True)
print(result)