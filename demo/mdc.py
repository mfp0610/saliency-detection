import math
import numpy as np
import cv2
import itertools

import utils.utils as utils

class MDCSaliency():
    def __init__(self, img):
        self.alpha, self.beta, self.theta = 0.8, 0.3, 0.
        self.original_map, self.aug_map = self._calculate_saliency(img)

    def _calculate_saliency(self, img):
        sum,sqsum = cv2.integral2(img)
        area_sum = lambda sum, x1, x2, y1, y2 : (sum[y2, x2, :] - sum[y1, x2, :] - sum[y2, x1, :] + sum[y1, x1, :])

        h, w = img.shape[0], img.shape[1]
        ret = np.zeros((h, w))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                tl = np.sum(area_sum(sqsum, 0, x + 1, 0, y + 1)) - 2 * np.sum(area_sum(sum, 0, x + 1, 0, y + 1) * img[y, x, :]) + (x + 1) * (y + 1) * np.sum(np.power(img[y, x, :], 2, dtype = np.uint32))
                tr = np.sum(area_sum(sqsum, x, -1, 0, y + 1)) - 2 * np.sum(area_sum(sum, x, -1, 0, y + 1) * img[y, x, :]) + (w - x) * (y + 1) * np.sum(np.power(img[y, x, :], 2, dtype = np.uint32))
                bl = np.sum(area_sum(sqsum, 0, x + 1, y, -1)) - 2 * np.sum(area_sum(sum, 0, x + 1, y, -1) * img[y, x, :]) + (x + 1) * (h - y) * np.sum(np.power(img[y, x, :], 2, dtype = np.uint32))
                br = np.sum(area_sum(sqsum, x, -1, y, -1)) - 2 * np.sum(area_sum(sum, x, -1, y, -1) * img[y, x, :]) + (w - x) * (h - y) * np.sum(np.power(img[y, x, :], 2, dtype = np.uint32))
                ret[y, x] = np.sqrt(np.min([tl, tr, bl, br]))
        ret = ret / np.max(ret) * 255
        
        gray  = ret.astype(np.uint8)
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        marker = np.zeros((img.shape[0], img.shape[1]), dtype = np.int32)
        marker = np.where(gray > (1 + self.theta) * t, 1, marker)
        marker = np.where(gray < (1 - self.theta) * t, 2, marker)
        marker = cv2.watershed(img, marker)
        ret_enhance = np.where(marker == 1, 1 - self.alpha * (1 - ret), ret)
        ret_enhance = np.where(marker == 2, self.alpha*ret, ret)

        return ret, ret_enhance
