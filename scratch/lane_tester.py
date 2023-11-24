import cv2
import matplotlib.pyplot as plt
from gelgenie.gel_tools import Preprocessor
from scratch.lane_finder import LaneFinder


if __name__ == '__main__':

    img = cv2.imread("../samples/image3.jpg")
    preprocessor = Preprocessor()
    img_grey = preprocessor.apply_grayscale(img)
    plt.imshow(img_grey, cmap='gray')
    plt.show()
    lane_finder = LaneFinder(img_grey)
    lanes = lane_finder.find_lanes()
    print(lanes)
    lane_finder.display_lanes(lanes)
