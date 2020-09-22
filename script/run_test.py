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
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)
    # --- networks --- #
    pix2Food = Pix2FoodModel(opt, trainConfig)

    # --- prepare dataset --- #
    img_size = trainConfig["training"]["img_size"]
    NUM_TRAIN = trainConfig["training"]["NUM_TRAIN"]
    NOISE_DIM = trainConfig["training"]["NOISE_DIM"]
    batch_size = trainConfig["training"]["batch_size"]

    food_dataset = UNetDataset(trainConfig["training"]["data_root"],
                               trainConfig["training"]["img_size"],
                               transform=T.ToTensor(),
                               test=True)
    print(len(food_dataset))
    # loader_train, loader_val = getTrainAndVal(food_dataset, NUM_TRAIN, batch_size)
    bs = 1
    loadar_food = DataLoader(food_dataset, batch_size=bs, collate_fn=food_dataset.collate_fn)

    # --- load model --- #
    # modelName = "pix2food-patch512.pkl"
    # modelPath = os.path.join("../checkpoints", modelName)
    pix2Food.netG.load_state_dict(torch.load("../checkpoints/netG-patch512.pkl"))
    

