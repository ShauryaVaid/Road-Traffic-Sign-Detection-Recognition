import cv2
import numpy as np


def preprocess_image(image):
    # Resize image to 32 × 32
    image = cv2.resize(image, (32, 32))

    # Convert pixel values from 0-255 to 0-1
    image = image / 255.0

    return image