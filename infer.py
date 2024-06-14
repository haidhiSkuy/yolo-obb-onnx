import cv2
import numpy as np 
from lib.nms import non_max_suppression
from lib.utils import scale_boxes, regularize_rboxes
from typing import Union
import onnxruntime as ort 
from lib.utils import letterbox 

class YoloOrientedBoxes: 
    def __init__(self, onnx_path): 
        self.onnx_path = onnx_path 
        self.session = ort.InferenceSession(self.onnx_path)  
        self.LETTERBOX_SHAPE = (1, 3, 1024, 1024)
        self.orig_image = None
    
    def __call__(self, image_arr: np.ndarray): 
        preprocessed = self.preprocessing(image_arr) 
        prediction = self.infer(preprocessed)
        result = self.postprocess(prediction)
        return result

    def preprocessing(self, image_arr: np.ndarray): 
        self.orig_image = image_arr.copy()
        input_img = letterbox(image_arr)
        img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
        img = np.transpose(img_rgb, (2, 0, 1)) 
        img = np.expand_dims(img, axis=0) 
        img = img.astype(np.float32) 
        return img

    def infer(self, image_input: np.ndarray): 
        output = self.session.run(None, {'images': image_input})[0]
        preds = non_max_suppression(output) 
        return np.expand_dims(preds[0], axis=1)

    def postprocess(self, output_arr: np.ndarray): 
        results = { 
            "boxes": [], 
            "class": [], 
            "scores": []
        }
        for pred in output_arr: 
            rboxes = regularize_rboxes(np.concatenate([pred[:, :4], pred[:, -1:]], axis=-1))
            rboxes[:, :4] = scale_boxes(
                self.LETTERBOX_SHAPE[2:], 
                rboxes[:, :4], 
                self.orig_image.shape, 
                xywh=True
            ) 

            results["boxes"].append(rboxes[0].tolist()) 
            results["class"].append(pred[0][5])
            results["scores"].append(pred[0][6])

        return results