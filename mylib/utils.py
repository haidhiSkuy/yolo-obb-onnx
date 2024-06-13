import cv2 
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


# def draw_bbox()