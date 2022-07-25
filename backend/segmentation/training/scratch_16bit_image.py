import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import cv2

filename = 'C:/2022_Summer_Intern/Gel_Images_UNet_Test/Images_Q1+Q2+selected/066.tif'
filename2 = 'C:/2022_Summer_Intern/Gel_Images_UNet_Test/Images_Q1+Q2+selected/UVP01940May142019.jpg'


ioimg = imageio.imread(filename2)
# for 8-bit images, max = 255, min = 0? 1?
# for 16-bit images, max = 65535, min = 0

# to normalize:  (value - min)/(max-min)
# z = np.tile(ioimg,(3,1,1))  # change to color

# 1 - see how to do the resize (change resample function)
# 2 - how can we convert to L correctly from 16-bit? Should we just convert to 8-bit first?
# 3 - does pytorch read in the 16-bit images correctly?

# color/ grayscale
if ioimg.shape[-1] == 3:  # color
    ioimg = cv2.cvtColor(ioimg, cv2.COLOR_RGB2GRAY)
else:  #greyscale
    pass


# no. of bits in image
if ioimg.dtype == 'uint8':
    max_val = 255
elif ioimg.dtype == 'uint16':
    max_val = 65535

ioimg = (ioimg.astype(np.float32)-max_val)/(max_val-0)  # Normalization


# there are 3 cases
# 1 - image is 8-bit/RGB - read in, convert to grayscale (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), convert to float, normalize to 0-1
# 2 - image is -bit/greyscale - as above without conversion
# 3 - 16-bit/gray - as above but make sure to use 65535 as max instead of 256
# 4 - 16-bit/rgb

plt.imshow(ioimg, cmap='Reds')  # greyscale -> red color map for viewing
plt.show()

# color
# z = z.transpose((1,2,0))
