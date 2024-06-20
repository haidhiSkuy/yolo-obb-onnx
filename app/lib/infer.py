import cv2
import numpy as np 
import onnxruntime as ort 
from lib.utils import letterbox 
from lib.nms import non_max_suppression
from lib.utils import scale_boxes, regularize_rboxes

class YoloOrientedBoxes: 
    def __init__(self, onnx_path: str): 
        self.onnx_path = onnx_path 
        self.session = ort.InferenceSession(self.onnx_path)  
        self.LETTERBOX_SHAPE = (1, 3, 1024, 1024)
        self.orig_image = None
        self.classes_dict = {
            0:'plane', 1:'ship', 2:'storage tank',3:'baseball diamond',4:'tennis court',
            5:'basketball court',6:'ground track field',7:'harbor',8:'bridge',9:'large vehicle',
            10:'small vehicle',11:'helicopter',12:'roundabout',13:'soccer ball field',14:'swimming pool'
        }
    
    def __call__(self, image_arr: np.ndarray) -> dict: 
        preprocessed = self.preprocessing(image_arr) 
        prediction = self.infer(preprocessed)
        result = self.postprocess(prediction)
        return result

    def preprocessing(self, image_arr: np.ndarray) -> np.ndarray: 
        self.orig_image = image_arr.copy()
        if image_arr.shape[2] == 4: #rgba 
            image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGBA2RGB)

        input_img = letterbox(image_arr)
        img = np.transpose(input_img, (2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0) 
        img = img.astype(np.float32) 
        return img

    def infer(self, image_input: np.ndarray) -> np.ndarray: 
        output = self.session.run(None, {'images': image_input})[0]
        preds = non_max_suppression(output) 
        return np.expand_dims(preds[0], axis=1)

    def postprocess(self, output_arr: np.ndarray) -> dict: 
        results = { 
            "boxes": [], 
            "classes": [], 
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

            out_boxes = rboxes[0].tolist()
            # out_boxes = list(map(int, rboxes[0].tolist()))
            out_class = self.classes_dict[pred[0][5]] 
            out_scores = round(pred[0][6], 2)

            results["boxes"].append(out_boxes) 
            results["classes"].append(out_class)
            results["scores"].append(out_scores)

        return results