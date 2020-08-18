import os
import pickle
import numpy as np
import yaml
from torchvision import transforms as T
import torch

from options.options import BaseOptions
from models.pix2food_model import Pix2FoodModel
from dataset.mydataset import OriginDataset, UNetDataset, getTrainAndVal
from models.utils import show_all_images, Tensor2Image

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
                               transform=T.ToTensor())
    print(len(food_dataset))
    loader_train, loader_val = getTrainAndVal(food_dataset, NUM_TRAIN, batch_size)

    modelName = "pix2food-patch512.pkl"
    modelPath = os.path.join(opt.checkpoints_dir, modelName)

    # --- training --- #
    iteration = 0
    for epoch in range(trainConfig["training"]["num_epochs"]):
        for startImg, actImg, trueImg, imgIDs in loader_train:
            if startImg.shape[0] != 4:
                break
            pixImg = torch.cat((startImg, actImg), 1)
            # --- model update step --- #
            pix2Food.feedInput(pixImg, trueImg)
            pix2Food.optimizeGAN()
            predImg = pix2Food.predict()
            # --- show result --- #
            if iteration % trainConfig["training"]["show_every"] == 0:
                print(f"Epoch: {epoch}, Itration: {iteration}")
                print(f"imgID = {imgIDs}")
                pix2Food.PrintLossLog()
                show_all_images([startImg, actImg, trueImg, predImg])
                # show_all_images([trueImg, predImg])
            iteration += 1
            # --- saving model --- #
            if iteration % trainConfig["training"]["save_every"] == 0:
                with open(modelPath, 'wb') as modelFile:
                    pickle.dump(pix2Food, modelFile)
                modelFile.close()
                print(f"save model to {modelPath} at iteration = {iteration}")
        if epoch == trainConfig["training"]["num_epochs"] * 0.8:
            pix2Food.setlmd(0)

    with open(modelPath, 'wb') as modelFile:
        pickle.dump(pix2Food, modelFile)
    modelFile.close()
    print(f"Finally save model to {modelPath}")


    # # --- load model --- #
    # with open(modelPath, 'rb') as modelFile:
    #     pix2Food = pickle.load(modelFile)

    # --- validation --- #
    # print("\n\n\n # --- Validation Set --- # \n\n\n")

    # for startImg, actImg, trueImg, imgIDs in loader_val:
    #     if startImg.shape[0] != 4:
    #         break
    #     # --- model update step --- #
    #     pixImg = torch.cat((startImg, actImg), 1)
    #     pix2Food.feedInput(pixImg, trueImg)
    #     predImg = pix2Food.predict()
    #     # --- show result --- #
    #     show_all_images([startImg, actImg, trueImg, predImg])
