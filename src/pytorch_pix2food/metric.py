import cv2
import numpy as np


def pixelAccuracy(gt, pd):
    if gt.shape[2] == 3 or pd.shape[2] == 3:
        gt, pd = gt[:, :, 0], pd[:, :, 0]
    n_00 = np.sum((gt == 0) * (pd == 0))
    t_0 = np.sum(gt == 0)
    n_11 = np.sum((gt == 255) * (pd == 255))
    t_1 = np.sum(gt == 255)
    # print(f"c = {t_0 + t_1}, t = {gt.shape[0] * gt.shape[1]}")
    pixAccuracy = (n_00 + n_11) / (t_0 + t_1)
    meanAccurracy = ((n_00 / t_0) + (n_11 / t_1)) / 2
    return pixAccuracy, meanAccurracy


def IoU(gt, pd):
    # class 0:
    i_0 = np.sum((gt == 0) * (pd == 0))
    u_0 = np.sum(((gt == 0) + (pd == 0)) > 0)
    # class 1:
    i_1 = np.sum((gt == 255) * (pd == 255))
    u_1 = np.sum(((gt == 255) + (pd == 255)) > 0)

    meanIoU = (i_0 / u_0 + i_1 / u_1) / 2

    return meanIoU
