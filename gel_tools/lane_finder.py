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

    def find_lanes(self):

        # Apply thresholding, try messing around with other thresholding methods
        ret, thresh = cv2.threshold(self.image, 90, 255, cv2.THRESH_BINARY)

        # Find contours, different methods/parameters might give better results
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        s_cnt = [cnt for cnt in contours if cv2.contourArea(cnt) > 200]

        contours = s_cnt

        tot = np.zeros([len(contours), 3])

        # extract the first x,y co-ordinates of each contour
        for k in range(0, len(contours)):
            tot[k] = ([min(contours[k][:, 0, 0]), max(contours[k][:, 0, 0]), k])

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
