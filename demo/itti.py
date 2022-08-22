import math
import numpy as np
import cv2
import itertools

import utils.utils as utils


class GaussianPyramid:
    # Gaussian Pyramid
    def __init__(self, src):
        self.maps = self.__make_gaussian_pyramid(src)

    def __make_gaussian_pyramid(self, src):
        # output a series of feature map
        maps = {'intensity': [],
                'colors': {'b': [], 'g': [], 'r': [], 'y': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        amax = np.amax(src)
        b, g, r = cv2.split(src)
        for x in range(1, 9):
            # smoothing and downsampling
            b, g, r = map(cv2.pyrDown, [b, g, r])
            if x < 2:
                continue
            buf_its = np.zeros(b.shape)
            buf_col = list(map(lambda _: np.zeros(b.shape), range(4)))
            for y, x in itertools.product(range(len(b)), range(len(b[0]))):
                # generate the intensity map and differential color difference map
                buf_its[y][x] = self.__get_intensity(b[y][x], g[y][x], r[y][x])
                buf_col[0][y][x], buf_col[1][y][x], buf_col[2][y][x], buf_col[3][y][x] = \
                    self.__get_colors(b[y][x], g[y][x], r[y][x], buf_its[y][x], amax)
            maps['intensity'].append(buf_its)
            for (color, index) in zip(sorted(maps['colors'].keys()), range(4)):
                maps['colors'][color].append(buf_col[index])
            for (orientation, index) in zip(sorted(maps['orientations'].keys()), range(4)):
                maps['orientations'][orientation].append(self.__conv_gabor(buf_its, np.pi * index / 4))
        return maps

    def __get_intensity(self, b, g, r):
        # get the intensity
        return (np.float64(b) + np.float64(g) + np.float64(r)) / 3.

    def __get_colors(self, b, g, r, i, amax):
        # get the differential color difference map
        b, g, r = list(map(lambda x: np.float64(x) if (x > 0.1 * amax) else 0., [b, g, r]))  # set < max/10 to 0
        nb, ng, nr = list(map(lambda x, y, z: max(x - (y + z) / 2., 0.), [b, g, r], [g, b, r], [r, g, b]))  # generate 3 maps
        ny = max(((r + g) / 2. - math.fabs(r - g) / 2. - b), 0.)  # the fourth map
        if i != 0.0:
            return list(map(lambda x: x / np.float64(i), [nb, ng, nr, ny]))
        else:
            return nb, ng, nr, ny

    def __conv_gabor(self, src, theta):
        # a linear filter for edge extraction
        kernel = cv2.getGaborKernel((8, 8), 4, theta, 8, 1)
        return cv2.filter2D(src, cv2.CV_32F, kernel)


class FeatureMap:
    # get the differential feature map
    def __init__(self,srcs):
        self.maps = self.__make_feature_map(srcs)

    def __make_feature_map(self, srcs):
        # calculate the index of center-surround
        cs_index = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 6)) 
        maps = {'intensity': [],
                'colors': {'bg': [], 'ry': []},
                'orientations': {'0': [], '45': [], '90': [], '135': []}}
        for c, s in cs_index:
            maps['intensity'].append(self.__scale_diff(srcs['intensity'][c], srcs['intensity'][s]))
            for key in maps['orientations'].keys():
                maps['orientations'][key].append(self.__scale_diff(srcs['orientations'][key][c], srcs['orientations'][key][s]))
            for key in maps['colors'].keys():
                maps['colors'][key].append(self.__scale_color_diff(
                    srcs['colors'][key[0]][c], srcs['colors'][key[0]][s],
                    srcs['colors'][key[1]][c], srcs['colors'][key[1]][s]
                ))
        return maps

    def __scale_diff(self, c, s):
        c_size = tuple(reversed(c.shape))
        return cv2.absdiff(c, cv2.resize(s, c_size, None, 0, 0, cv2.INTER_NEAREST))

    def __scale_color_diff(self,c1,s1,c2,s2):
        c_size = tuple(reversed(c1.shape))
        return cv2.absdiff(c1 - c2, cv2.resize(s2 - s1, c_size, None, 0, 0, cv2.INTER_NEAREST))


class ConspicuityMap:
    def __init__(self, srcs):
        self.maps = self.__make_conspicuity_map(srcs)

    def __make_conspicuity_map(self, srcs):
        intensity = self.__scale_add(list(map(utils.normalize, srcs['intensity'])))
        for key in srcs['colors'].keys():
            srcs['colors'][key] = list(map(utils.normalize, srcs['colors'][key]))
        color = self.__scale_add([srcs['colors']['bg'][x] + srcs['colors']['ry'][x] for x in range(len(srcs['colors']['bg']))])  # 颜色累加
        orientation = np.zeros(intensity.shape)
        for key in srcs['orientations'].keys():
            orientation += self.__scale_add(list(map(utils.normalize, srcs['orientations'][key])))
        return {'intensity': intensity, 'color': color, 'orientation': orientation}

    def __scale_add(self, srcs):
        buf = np.zeros(srcs[0].shape)
        for x in srcs:
            buf += cv2.resize(x, tuple(reversed(buf.shape)))
        return buf


class ITTISaliencyMap:
    # saliency map
    def __init__(self, src):
        self.gp = GaussianPyramid(src)
        self.fm = FeatureMap(self.gp.maps)
        self.cm = ConspicuityMap(self.fm.maps)
        self.map = cv2.resize(self.__make_saliency_map(self.cm.maps), tuple(reversed(src.shape[0:2])))

    def __make_saliency_map(self, srcs):
        srcs = list(map(utils.normalize, [srcs[key] for key in srcs.keys()]))
        return srcs[0] / 3. + srcs[1] / 3. + srcs[2] / 3.