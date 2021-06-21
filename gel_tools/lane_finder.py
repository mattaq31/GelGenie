import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict


class LaneFinder:
    def __init__(self, image):
        self.image = image

    def display_lanes(self, lanes, figsize=(20, 20)):
        plt.figure(figsize=figsize)
        plt.imshow(self.image, cmap='gray')
        for v in lanes.keys():
            plt.axvline(lanes[v][0], 0, color='b')
            plt.axvline(lanes[v][1], 0, color='r')
        plt.show()


    # Function is a work in progress
    def split_bands(self, thresh):
        """Split distinct bands which would otherwise overlap and
        be detected as one contour."""

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find sure background area with dilation
        sure_bg = cv2.dilate(opening, kernel, iterations=1)

        # Find sure foreground area with erosion
        kernel = np.ones((20, 20), np.uint8)
        sure_fg = cv2.erode(opening,kernel,iterations=1)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Image plots for testing
        plt.figure()
        # plt.imshow(sure_fg)
        plt.imshow(unknown)

        # Label each individual foregroup shape with a different integer
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # plt.imshow(markers)
        # Convert image to BGR color
        water_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Apply watershed algorithm
        markers = cv2.watershed(water_image, markers)
        water_image[markers == -1] = [255, 0, 0]
        plt.imshow(markers)
        return markers

    def find_lanes(self):
        # Apply thresholding, try messing around with other thresholding methods
        # ret, thresh = cv2.threshold(self.image, 90, 255, cv2.THRESH_BINARY)

        # Find contours, different methods/parameters might give better results
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



        # Adding morphology to break up contours
        # TODO: Test how well this generalises to other pictures

        # Apply thresholding, try messing around with other thresholding methods
        ret, thresh = cv2.threshold(self.image, 90, 255,
                                    cv2.THRESH_BINARY)

        # Visualize thresholding
        plt.figure()
        plt.imshow(thresh, cmap="gray")

        # Split bands which are overlapping
        markers = self.split_bands(thresh)
        # Bypass split_bands() for testing (disable one or the other)
        # markers = thresh

        # Convert the 32-bit image from split_bands() to 8-bit
        img_8bit = markers.astype(np.uint8)

        # Attempt to threshold again
        ret, thresh2 = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # Show image
        plt.figure()
        plt.imshow(thresh2)

        # Find contours, different methods/parameters might give better results
        contours, hierarchy = cv2.findContours(thresh2,
                                               cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # print(contours)

        # color_img = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        cnt = cv2.drawContours(self.image, contours, -1, (0, 255, 0), 3)

        # Show image
        plt.figure()
        plt.imshow(cnt)

        # Eliminate small contours which are not bands
        s_cnt = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

        contours = s_cnt
        # print(contours)


        tot = np.zeros([len(contours), 3])

        # extract the first x,y co-ordinates of each contour
        for k in range(0, len(contours)):
            tot[k] = ([min(contours[k][:, 0, 0]), max(contours[k][:, 0, 0]), k])

        print(tot)

        # sort the max. x,y co-ordinates of each contour in ascending order whilst linking the contour number
        ind3 = np.argsort(tot, axis=0)
        tot_sorted = tot[ind3[:, 0], :]

        col = 0

        lanes = {}
        maxedge = tot_sorted[0, 1]
        minedge = tot_sorted[0, 0]

        # Find the positions of the different lanes
        for x in range(0, len(tot_sorted)-1):

            if tot_sorted[x, 1] <= tot_sorted[x+1, 0]:
                lanes[col] = (minedge, maxedge)
                maxedge = tot_sorted[x+1, 1]
                minedge = tot_sorted[x+1, 0]
                col = col + 1
                # if x == len(tot_sorted)-2:
                #     lanes[col] = (minedge, maxedge)
            else:
                maxedge = max(maxedge, tot_sorted[x+1, 1])
                minedge = min(minedge, tot_sorted[x+1, 0])

        lanes[col] = (minedge, maxedge)
        # plt.imshow(self.image, cmap='gray')
        #
        # for v in range(0, col+1):
        #     plt.axvline(lanes[v][0], 0, color='b')
        #     plt.axvline(lanes[v][1], 0, color='r')

        return lanes
        # TODO: Cleanup code and return lane locations in a more orderly fashion
        # TODO: Deal with inverted images
        # TODO: Additional class for gel image?
        # TODO: Get rid of contour area parameters (or fine-tune)
