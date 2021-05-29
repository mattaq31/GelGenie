import cv2


class Preprocessor:
    def __init__(self):
        pass

    def apply_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


