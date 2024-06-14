import torch 
import numpy as np 
import cv2 
import time
import math 


def _get_covariance_matrix(boxes):
    gbbs = np.concatenate((np.square(boxes[:, 2:4]) / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    cos = np.cos(c)
    sin = np.sin(c)
    cos2 = np.square(cos)
    sin2 = np.square(sin)
    cov_xx = a * cos2 + b * sin2
    cov_yy = a * sin2 + b * cos2
    cov_xy = (a - b) * cos * sin
    return cov_xx, cov_yy, cov_xy

def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)


    t1 = (((a1 + a2.T) * (y1 - y2.T)**2 + (b1 + b2.T) * (x1 - x2.T)**2) / ((a1 + a2.T) * (b1 + b2.T) - (c1 + c2.T)**2 + eps)) * 0.25
    t2 = (((c1 + c2.T) * (x2.T - x1) * (y1 - y2.T)) / ((a1 + a2.T) * (b1.T + b2) - (c1 + c2.T)**2 + eps)) * 0.5
    t3 = np.log(((a1.T + a2) * (b1.T + b2) - (c1 + c2.T)**2) 
        / (4 * np.sqrt((np.maximum(a1 * b1 - c1**2, 0)) * (np.maximum(a2.T * b2.T - c2.T**2, 0))) + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd

def nms_rotated(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)
    keep = np.where(ious.max(axis=0) < threshold)[0]
    return sorted_idx[keep]

def xywh2xyxy(x):
    y = np.empty_like(x) if isinstance(x, np.ndarray) else np.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.3,
    iou_thres=0.7,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=15, 
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=True,
):
   
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = np.transpose(prediction, (0, 2, 1))  # shape(1,84,6300) to shape(1,6300,84)
   
    t = time.time()
    output = [np.zeros((0, 6 + nm), dtype=np.float32) for _ in range(bs)]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
 
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        split_points = [4, 4 + nc, 4 + nc + nm]

        box = x[:, :split_points[0]]
        cls = x[:, split_points[0]:split_points[1]]
        mask = x[:, split_points[1]:]


        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j][:, np.newaxis], j[:, np.newaxis].astype(float), mask[i]), axis=1)
        else:  # best class only
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1)
            x = np.concatenate((box, conf, j[:, np.newaxis].astype(float), mask), axis=1)
            x = x[conf.flatten() > conf_thres]
        
        
        # Filter by class
        if classes is not None:
            mask = np.any(x[:, 5:6] == classes, axis=1)
            x = x[mask]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

     
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores

        boxes = np.concatenate([x[:, :2] + c, x[:, 2:4], x[:, -1:]], axis=-1)  # xywhr
        i = nms_rotated(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
    return output