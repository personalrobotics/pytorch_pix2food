import os
import pickle
import numpy as np
import yaml
from torchvision import transforms as T
import torch

from options.options import BaseOptions
from models.pix2food_model import Pix2FoodModel
from dataset.mydataset import OriginDataset, UNetDataset, getTrainAndVal
from models.utils import show_all_images, Tensor2Image, save_images, save_images2
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # --- opt --- #
    opt = BaseOptions().parse()   # get training options
    # print(vars(opt))

    # --- config --- #
    configPath = "cGAN_config.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)
    # --- networks --- #
    # pix2Food = Pix2FoodModel(opt, trainConfig)

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
    bs = 2
    loadar_food = DataLoader(food_dataset, bs)

    modelName = "pix2food-patch512.pkl"
    modelPath = os.path.join(opt.checkpoints_dir, modelName)

    # --- load model --- #
    with open(modelPath, 'rb') as modelFile:
        pix2Food = pickle.load(modelFile)

    # --- validation --- #
    print("\n\n\n # --- Validation Set --- # \n\n\n")
    iteration = 0
    for startImg, actImg, trueImg, imgIDs in loadar_food:
        if startImg.shape[0] != bs:
            break
        # --- model update step --- #
        pixImg = torch.cat((startImg, actImg), 1)
        pix2Food.feedInput(pixImg, trueImg)
        predImg = pix2Food.predict()
        # --- show result --- #
        show_all_images([startImg, actImg, trueImg, predImg])
        print(predImg.shape)
        # save_images([predImg], str(iteration))
        save_images2([predImg], imgIDs)
        iteration += 1

