import cv2
import torch 
import numpy as np 
from mylib.nms import *
from typing import Union
import onnxruntime as ort 
import matplotlib.pyplot as plt
from mylib.utils import letterbox

def preprocessing(image: Union[str, np.ndarray]): 
    input_img = cv2.imread(image) if isinstance(image, str) else image  
    input_img = letterbox(input_img) 
    img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.transpose(img_rgb, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0) 
    img = img.astype(np.float32) 
    return img

# Mine 
ort_session = ort.InferenceSession("/workspaces/obb/yolov8n-obb.onnx")  

img = cv2.imread("sample.jpg")
output = ort_session.run(None, {'images': preprocessing(img)})[0]

input_tensor = torch.tensor(output) 
preds = non_max_suppression(input_tensor)



sample = letterbox(img) 
for i in preds[0].numpy(): 
    center_x, center_y, width, height, angle, confidence, obj_class = i
    angle = np.degrees(angle)
    rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box) 
    cv2.drawContours(sample, [box], 0, (0, 255, 0), 2)


cv2.imwrite("result.jpg", sample)