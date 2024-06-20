import cv2 
import numpy as np
from PIL import Image 
from io import BytesIO
from lib.infer import YoloOrientedBoxes 
from fastapi import FastAPI, UploadFile, Response

app = FastAPI(title="Oriented Bounding Box")
obb = YoloOrientedBoxes("model/yolov8n-obb.onnx")

def load_image(data):
    return Image.open(BytesIO(data))

@app.get("/")
async def hello():
    return {"Hello": "World"}

@app.post("/obb")
async def iference(image: UploadFile):
    contents = await image.read() 
    input_image = load_image(contents)
    input_image = np.array(input_image)
    output = obb(input_image)

    return output
