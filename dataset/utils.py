import time
import cv2
import numpy as np
from numpy.linalg import norm

##############################################################################
# decorator
##############################################################################


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def kmeans(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    roi = img[ymin:ymax, xmin:xmax]
    Z = roi.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    # print(f"label = {label.shape}, center = {center.shape}")
    center = np.uint8(center * 255)
    blue = np.array([23, 66, 109])
    light = np.array([255] * 3)
    dark = np.array([0] * 3)
    # if norm(center[0] - blue) < norm(center[1] - blue):
    #     center = np.array([dark, light])
    # else:
    #     center = np.array([light, dark])
    print(f"center={center}")
    res = center[label.flatten()].reshape((roi.shape))
    kmPixImg = np.zeros_like(img)
    kmPixImg[ymin:ymax, xmin:xmax] = res
    return res
