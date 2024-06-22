
# Onnxruntime & FastAPI YOLO Oriented Box

How to use (don't forget to download the onnx model)

```console 
user@ubuntu:~$ git clone git@github.com:widyamsib/widya-OBB.git 

user@ubuntu:~$ cd widya-OBB && docker compose up
```` 

Using The API 
```python
image_path = "image.jpg"

headers = {'accept': 'application/json'}

files = {'image': (image_path, open(image_path, 'rb'), 'image/jpeg')} 
    
response = requests.post('http://0.0.0.0:1234/obb', headers=headers, files=files)
output = response.json()
````

Draw Boxes
```python
import cv2

def draw_bbox(
        out_tensor : Union[list, np.ndarray], 
        image : np.ndarray, 
        line_thick : int = 1, 
        line_color : tuple = (0,255,0)
    ) -> np.ndarray: 
    center_x, center_y, width, height, angle = out_tensor
    angle = np.degrees(angle)
    rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box) 
    drawed = cv2.drawContours(image, [box], 0, line_color, line_thick)
    return drawed


img_arr = cv2.imread(image_path) 
for box in output['boxes']:
    drawed = draw_bbox(box, img_arr, line_thick=2, line_color=(100,10,255))
```
