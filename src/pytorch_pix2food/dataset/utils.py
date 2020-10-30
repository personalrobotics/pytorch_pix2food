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
    res = center[label.flatten()].reshape((roi.shape))
    kmPixImg = np.zeros_like(img)
    kmPixImg[ymin:ymax, xmin:xmax] = res
    return res

def generateActionImg(start, end, actImg=None, push_direction="left_push", img_size = (640, 480), forque_width=45):
    INTENSITY = 255
    r = forque_width // 2

    if actImg is None:
        actImg = np.zeros((img_size[1], img_size[0]))
    
    if push_direction in ["left_push", "right_push"]:
        distance_in_pixel = np.abs(end[0] - start[0])
    else:
        distance_in_pixel = np.abs(end[1] - start[1])

    x, y = start
    for i in range(distance_in_pixel):
        actionValue = INTENSITY * i / distance_in_pixel
        if push_direction == "left_push":
            actImg[y - r: y + r, x - i] = actionValue
        if push_direction == "right_push":
            actImg[y - r: y + r, x + i] = actionValue
        if push_direction == "up_push":
            actImg[y - i, x - r : x + r] = actionValue
        if push_direction == "down_push":
            actImg[y + i, x - r : x + r] = actionValue
    cv2.normalize(actImg, actImg, 0, 255, cv2.NORM_MINMAX)

    return actImg
