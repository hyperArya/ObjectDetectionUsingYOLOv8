#this code will not run in kaggle and google colab notebook because they don't support cv
#jupyter notebook recommended
#use 'pip install ultralytics' to install the required libraries
#after installing all the required libraries Restart Kernel
from ultralytics import YOLO
import cv2

#download the best.onnx or best.pt and change the file path
model = YOLO(r"C:\Users\arshd\Downloads\best.onnx")

result = model.predict (source="0", show=True)
print(result)
