from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from PIL import Image
import numpy as np
import cv2 as cv
import uvicorn
import torch
import io
import os

from ultralytics import YOLO

# Initialize application
app = FastAPI(title="Mask Detection deployed on FastAPI")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model 
    model = YOLO('./pretrained/best.pt')
    model.to(device)
    print(f'Loaded Model in {model.device}')

@app.on_event("startup")
async def get_model():
    load_model()
    
@app.get("/healthcheck/")
async def healthcheck():
    return 'Health - OK'

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    
    # Get predictions
    results = model(image)
    out_image = results[0].plot()
    out_image = cv.cvtColor(out_image, cv.COLOR_BGR2RGB)
    
    # predicted_image = obtain_image(image)
    pil_image = Image.fromarray(out_image.astype('uint8'), 'RGB')
    
    memory_stream = io.BytesIO()
    pil_image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
        
    # Prepare the response JSON object
    response_data = {
        "predictions": [],
        "image": f"data:image/png;base64,{filename}"
    }

    for box in results[0].boxes:
        if box.conf < 0.7:
            continue
        
        box = box.cpu().numpy() 
        response_data["predictions"].append({
            "class": results[0].names[int(box.cls[0])],
            "confidence": np.round(float(box.conf), 3),
            "bounding_box": {
                "x": int(box.xywh[0].astype(int)[0]),
                "y": int(box.xywh[0].astype(int)[1]),
                "width": int(box.xywh[0].astype(int)[2]),
                "height": int(box.xywh[0].astype(int)[3])
            }
        })
    
    #  return StreamingResponse(memory_stream, media_type="image/png")
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('APP_PORT')))
    uvicorn.run("yolo_app:app", reload=False, host="10.10.5.17", port=8080)
    # curl -X POST "http://10.10.5.17:8080/uploadfile/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@mask.png"
