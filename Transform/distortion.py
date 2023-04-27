from .interpolation import interpolation
from dip import *
import numpy as np
import math

class Distort:
    def __init__(self):
        pass

    def distortion(self, image, k):
        """Applies distortion to the image
                image: input image
                k: distortion Parameter
                return the distorted image"""

        res = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                cx = image.shape[0]/2
                cy = image.shape[1]/2
                
                ic = i - cx
                jc = j - cy
                r = math.sqrt(ic**2 + jc**2)
                icd = (1 / (1 + k * r)) * ic
                jcd = (1 / (1 + k * r)) * jc
                id = int(icd + cx)
                jd = int(jcd + cy)
                res[id][jd] = image[i][j]
        return res

    def correction_naive(self, distorted_image, k):
        """Applies correction to a distorted image by applying the inverse of the distortion function
        image: the input image
        k: distortion parameter
        return the corrected image"""

        res = np.zeros((distorted_image.shape[0]*2, distorted_image.shape[1]*2, distorted_image.shape[2]))
        for id in range(distorted_image.shape[0]):
            for jd in range(distorted_image.shape[1]):
                cx = distorted_image.shape[0]/2
                cy = distorted_image.shape[1]/2
                
                icd = id - cx
                jcd = jd - cy
                r = math.sqrt(icd**2 + jcd**2)
                ic = (1 + k * r) * icd
                jc = (1 + k * r) * jcd
                i = int(ic + cx)
                j = int(jc + cy)
                res[i][j] = distorted_image[id][jd]
        
        corrected = np.zeros(distorted_image.shape)
        for i in range(distorted_image.shape[0]):
            for j in range(distorted_image.shape[1]):
                corrected[i][j] = res[i][j]
        return corrected

    def correction(self, distorted_image, k, interpolation_type):
        """Applies correction to a distorted image and performs interpolation
                image: the input image
                k: distortion parameter
                interpolation_type: type of interpolation to use (nearest_neighbor, bilinear)
                return the corrected image"""

        inter = interpolation()
        res = np.zeros(distorted_image.shape)
        for i in range(distorted_image.shape[0]):
            for j in range(distorted_image.shape[1]):
                cx = distorted_image.shape[0]/2
                cy = distorted_image.shape[1]/2
                
                ic = i - cx
                jc = j - cy
                r = math.sqrt(ic**2 + jc**2)
                icd = (1 / (1 + k * r)) * ic
                jcd = (1 / (1 + k * r)) * jc
                id = round(icd + cx)
                jd = round(jcd + cy)
                res[i][j] = distorted_image[id][jd]
                if interpolation_type == "bilinear":
                    x1 = id - 1
                    x2 = id + 1
                    y1 = jd
                    y2 = jd + 1
                    q11 = distorted_image[x1][y1]
                    q12 = distorted_image[x1][y2]
                    q21 = distorted_image[x2][y1]
                    q22 = distorted_image[x2][y2]
                    x = id
                    y = jd
                    res[i][j] = inter.bilinear_interpolation(q11, q12, q21, q22, x, y, x1, x2, y1, y2)
        return res
