import os
import pickle
import numpy as np
import yaml
from torchvision import transforms as T
import torch
import cv2

from options.options import BaseOptions
from models.pix2food_model import Pix2FoodModel
from dataset.mydataset import OriginDataset, UNetDataset, getTrainAndVal
from dataset.utils import kmeans
from models.utils import show_all_images, show_all_images_rotate, Tensor2Image, save_images, save_images2
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from metric import pixelAccuracy, IoU

if __name__ == '__main__':
    # --- opt --- #
    opt = BaseOptions().parse()   # get training options
    # print(vars(opt))

    # --- config --- #
    configPath = "cGAN_config.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file)
        # trainConfig = yaml.load(file, Loader=yaml.FullLoader)
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
    bs = 1
    loadar_food = DataLoader(food_dataset, batch_size=bs, collate_fn=food_dataset.collate_fn)

    modelName = "pix2food-patch512.pkl"
    modelPath = os.path.join(opt.checkpoints_dir, modelName)
    # --- load model --- #
    with open(modelPath, 'rb') as modelFile:
        pix2Food = pickle.load(modelFile)

    # --- validation --- #
    print("\n\n\n # --- Validation Set --- # \n\n\n")
    iteration = 0
    for startImg, actImg, trueImg, imgIDs, plateBBox in loadar_food:
        if startImg.shape[0] != bs:
            break
        # --- model update step --- #
        # print(f"plateBBox = {plateBBox[0]}")
        # print(f"imgIDs = {imgIDs}")
        pixImg = torch.cat((startImg, actImg), 1)
        pix2Food.feedInput(pixImg, trueImg)
        predImg = pix2Food.predict()
        # --- show result --- #
        show_all_images_rotate([startImg, actImg, trueImg, predImg])
        trueImg, predImg = Tensor2Image(trueImg[0]), Tensor2Image(predImg[0])
        trueImg, predImg = cv2.resize(trueImg, (640, 480)), cv2.resize(predImg, (640, 480))
        # kmTrueImg = kmeans(trueImg, plateBBox[0])
        # kmPredImg = kmeans(predImg, plateBBox[0])
        # plt.imshow(kmTrueImg)
        # plt.show()
        # plt.imshow(kmPredImg)
        # plt.show()
        # save_images([predImg], str(iteration))
        # save_images2([predImg], imgIDs)

        # Metric:
        # pixAccuracy, meanAccuracy = pixelAccuracy(kmTrueImg, kmPredImg)
        # meanIoU = IoU(kmTrueImg, kmPredImg)
        # print(f"pixAcurracy = {pixAccuracy}, meanAccuracy = {meanAccuracy}, meanIoU = {meanIoU}")
        iteration += 1
