import cv2 
import math
import numpy as np


def letterbox(
        image, 
        target_width=1024, 
        target_height=1024, 
        color=(128, 128, 128)
    ):
    original_height, original_width = image.shape[:2]
    
    # Calculate the new dimensions preserving the aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create a new image with the target dimensions and fill it with the padding color
    letterbox_image = np.full((target_height, target_width, 3), color, dtype=np.uint8)
    
    # Calculate the top-left corner where the resized image will be placed
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    
    # Place the resized image onto the letterbox image
    letterbox_image[top:top+new_height, left:left+new_width] = resized_image
    
    return letterbox_image


def draw_bbox(out_tensor, image): 
    center_x, center_y, width, height, angle = out_tensor
    angle = np.degrees(angle)
    rect = ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box) 
    drawed = cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    return drawed

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape) 

def regularize_rboxes(rboxes):
    x, y, w, h, t = np.split(rboxes, 5, axis=-1)
    # Swap edge and angle if h >= w
    w_ = np.where(w > h, w, h)
    h_ = np.where(w > h, h, w)
    t = np.where(w > h, t, t + math.pi / 2) % math.pi
    return np.concatenate([x, y, w_, h_, t], axis=-1)  # regularized boxes 