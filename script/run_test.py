#!/usr/bin/env python

import os
import pickle
import numpy as np
import yaml
from torchvision import transforms as T
import torch
import cv2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from pytorch_pix2food.options.options import BaseOptions
from pytorch_pix2food.models.pix2food_model import Pix2FoodModel
from pytorch_pix2food.dataset.mydataset import OriginDataset, UNetDataset, getTrainAndVal
from pytorch_pix2food.dataset.utils import kmeans
from pytorch_pix2food.models.utils import show_all_images, show_all_images_rotate, Tensor2Image, save_images, save_images2
from pytorch_pix2food.metric import pixelAccuracy, IoU

if __name__ == '__main__':
    # --- opt --- #
    opt = BaseOptions().parse()   # get training options
    # print(vars(opt))

    # --- config --- #
    configPath = "cGAN_config.yaml"
    with open(configPath, 'rb') as file:
        # trainConfig = yaml.load(file, Loader=yaml.FullLoader)
        trainConfig = yaml.load(file)
    print(trainConfig)
    pix2Food = Pix2FoodModel(opt, trainConfig)
    img_size = trainConfig["training"]["img_size"]
    pix2Food.netG.load_state_dict(torch.load("../checkpoints/netG-patch512.pkl"))
    print(pix2Food)

