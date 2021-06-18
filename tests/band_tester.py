import cv2
import matplotlib.pyplot as plt
from gel_tools.utils import Preprocessor
from gel_tools.band_measure import Band


if __name__ == '__main__':

    img = cv2.imread("../samples/image3.tif")
    preprocessor = Preprocessor()
    img_grey = preprocessor.apply_grayscale(img)
    band_measure = Band()
    profile = band_measure.build_profile(img_grey, (258, 303))
