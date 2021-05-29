import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.signal import find_peaks, peak_widths


class Band:
    def __init__(self):
        self.raw_profile = []
        self.profile = []
        self.peak_widths = []
        self.backgrnd = []
        self.band_percentages = []

    def build_profile(self, image, lane):

        image = copy.copy(image)

        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.axvline(lane[0], 0, color='b')
        plt.axvline(lane[1], 0, color='r')
        plt.show()

        profile = np.mean(image[:, lane[0]:lane[1]], axis=1)
        plt.figure()
        plt.plot(profile)
        plt.show()
        self.raw_profile = profile
        self.profile = profile

    def find_peaks(self):
        peaks, _ = find_peaks(self.raw_profile,prominence=50)
        self.peak_widths = peak_widths(self.raw_profile, peaks, rel_height=0.9)

    def remove_background(self):
        start = 0
        for i in range(0, len(self.peak_widths[0])):
            s_peak = self.peak_widths[2][i].astype(int)
            self.backgrnd.extend(range(start, s_peak))
            start = self.peak_widths[3][i].astype(int)
        self.backgrnd.extend(range(start, len(self.raw_profile)))
        self.profile = self.raw_profile - np.mean(np.take(self.raw_profile, self.backgrnd))

    def quantify_bands(self):
        band_quant = []
        for i in range(0, len(self.peak_widths[0])):
            s_peak = self.peak_widths[2][i].astype(int)
            f_peak = self.peak_widths[3][i].astype(int)
            band_quant.append(sum(self.profile[s_peak:f_peak]))

        self.band_percentages = [band/sum(band_quant) for band in band_quant]
