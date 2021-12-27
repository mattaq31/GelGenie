import cv2
import io
import base64
import numpy as np


def convert_pil_image_base_64(*images, im_format='PNG'):
    b64_images = []
    for image in images:
        buff = io.BytesIO()
        image.save(buff, format=im_format)
        b64_images.append(base64.b64encode(buff.getvalue()).decode("ascii"))
    return b64_images


def convert_numpy_image_base_64(*images, im_format='png', max_intensity=1.0):
    b64_images = []
    converter = 65535/max_intensity
    for image in images:
        _, imagebytes = cv2.imencode('.%s' % im_format, np.uint16(image * converter))
        b64_images.append(base64.b64encode(imagebytes).decode('utf-8'))
    return b64_images


class Preprocessor:
    def __init__(self):
        pass

    def apply_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


