from gel_tools.band_finder import find_bands, load_image, load_image_b64
from gui.lane_finder import LaneFinder
from PIL import Image
import numpy as np
from skimage.util import img_as_uint
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img_1 = Image.open('/Users/matt/Documents/PhD/research_output/Automatic_Gel_Analyzer/analysis/NEB_1KB_quick_load_plus.png')
img = np.array(img_1)
# Read image
# img = image.imread(path)
# Convert to grayscale
img = rgb2gray(img)
# Convert to a 16-bit 2D array (values from 0 to 65535)
img = img_as_uint(img)

width, height = img_1.size


fg = 32
bg = 26
rep = 2

result = find_bands(img, int(fg)/100*65535, int(bg)/100*65535, int(rep)) # 90*255, 50*255, 2
plt.show()
