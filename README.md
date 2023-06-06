# Machine Learning Deployment

Using YOLOv8 to get prediction for simple mask detection problem using FastAPI. Also trying out the deployment in Docker and Kubernetes. 

## Training YOLOv8

Install latest version of `ultralytics` for training yolov8. Download the **Face Mask Detection** from Kaggle from this [link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) to [`src`](src/). With some preprocessing the `.xml` annotation files into proper yolov8 labels format, we can train the model as follow (scripts in [`src`](src/)). 

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')

# Train the model
model.train(data='./datasets/face-mask-detection/data.yaml', epochs=100, imgsz=640, optimizer='AdamW')
```
## Run FastAPI

To check if the FastAPI is working on with the trained model.

```bash
python3 src/yolo_app.py
```

Go to swagger documentation built within FastAPI and try the model with test images to get Image response.
