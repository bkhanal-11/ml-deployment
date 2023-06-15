# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /opt/app

# Copy the current directory contents into the container at /app
COPY . /opt/app

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r dev-requirements.txt

# Make port 8080 available for internal testing
EXPOSE 8080

# Make port 3000 available for external use
EXPOSE 3000

# Run yolo_app.py when the container launches
CMD ["python3", "/opt/app/src/yolo_app.py"]
