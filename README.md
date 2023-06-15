# Machine Learning Deployment

This project uses YOLOv8 to perform simple mask detection using FastAPI, and tests deployment using Docker and Kubernetes.

## YOLOv8

YOLOv8 is a state-of-the-art object detection algorithm. It is a popular choice for computer vision tasks such as object detection, image segmentation, and more.
Project Structure

Download the **Face Mask Detection** from Kaggle from this [link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) to [`src`](src/). With some preprocessing the `.xml` annotation files into proper yolov8 labels format, we can train the model as follow (scripts in [`src`](src/yolo_train.py)). 

The project structure is as follows:

```bash
.
├── datasets
│   └── face-mask-detection
│       ├── labels
│       ├── images
│       └── data.yaml
├── kubernetes
│   ├── deployment.yaml
│   └── service.yaml
├── tests
│   ├── test_yolo_app.py
│   └── ...
├── src
│   ├── yolo_app.py
│   ├── yolo_train.py
│   ├── preprocessing.py
│   └── ...
├── pretrained
│   ├── best.pt
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── dev-requirements.txt
└── ...
```

`datasets`: Contains the dataset used to train the YOLOv8 model.

`kubernetes`: Contains the Kubernetes deployment and service files for the application.

`src`: Contains the source code for the FastAPI application and the YOLOv8 model.

`test`: Contains test code for FastAPI application.

`pretrained`: Contains pretrained YOLOv8 model on face mask detection.

`Dockerfile`: Defines the Docker image for the application.

`docker-compose.yml`: Defines the Docker Compose configuration for the application.

`requirements.txt`: Contains the Python dependencies required to run the application.

`dev-requirements.txt`: Contains the Python dependencies required to help in development.

## Installation

To install the dependencies required to run the application, run the following command:

```bash
pip3 install -r requirements.txt
pip3 install -r dev-requirements.txt
```

## Running the Application

To run the FastAPI application, run the following command:

```bash
python3 src/yolo_app.py
```

You can then access the Swagger documentation at `http://localhost:8080/docs`. You can also test the model using curl:

```bash
curl -X POST "http://0.0.0.0:8080/uploadfile/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@mask.png"
```

## Deploying the Application using Docker

To deploy the application using Docker, run the following command:

```bash
docker-compose up -d --build
```

This will build the Docker image and start the container. The application will be available at `http://localhost:8080`.

## Deploying the Application using Kubernetes

To deploy the application using Kubernetes, follow these steps:

1. Create a Kubernetes deployment using the `deployment.yaml` file in the kubernetes directory.
2. Create a Kubernetes service using the `service.yaml` file in the kubernetes directory.

The application will be available at the IP address of the Kubernetes service.

## Running Tests

To run tests for the application, run the following command:

```bash
pytest
```
This will run all the tests in the tests directory.

## Resources

- [YOLOv8 documentation](https://docs.ultralytics.com/)

- [FastAPI documentation](https://fastapi.tiangolo.com/tutorial/)

- [Docker documentation](https://docs.docker.com/)

- [Kubernetes documentation](https://kubernetes.io/docs/home/)

- [FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)

- [Docker Tutorial for Beginners - A Full DevOps Course on How to Run Applications in Containers](https://www.youtube.com/watch?v=fqMOX6JJhGo)

- [Kubernetes Course - Full Beginners Tutorial (Containerize Your Apps!)](https://www.youtube.com/watch?v=d6WC5n9G_sM)

- [How to run FastAPI app on multiple ports?](https://stackoverflow.com/a/69641645)
