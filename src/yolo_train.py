from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2 as cv

# Load a model
model = YOLO('yolov8n.yaml')

# Train the model
model.train(data='./datasets/face-mask-detection/data.yaml', epochs=100, imgsz=640, optimizer='AdamW')

vis_result = False

if vis_result:
    results = model('./datasets/face-mask-detection/images/test/maksssksksss0.png')
    res_plotted = results[0].plot()
    plt.imshow(cv.cvtColor(res_plotted, cv.COLOR_BGR2RGB))
    plt.show()
    