import os
import pickle
import numpy as np
import yaml
from torchvision import transforms as T

from options.options import BaseOptions
from models.pix2food_model import Pix2FoodModel
from dataset.mydataset import RFDataset, getTrainAndVal
from models.utils import show_all_images

if __name__ == '__main__':
    # --- opt --- #
    opt = BaseOptions().parse()   # get training options
    # print(vars(opt))

    # --- config --- #
    configPath = "config.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)
    # --- networks --- #
    pix2Food = Pix2FoodModel(opt, trainConfig)

    # --- prepare dataset --- #
    img_size = trainConfig["training"]["img_size"]
    NUM_TRAIN = trainConfig["training"]["NUM_TRAIN"]
    NOISE_DIM = trainConfig["training"]["NOISE_DIM"]
    batch_size = trainConfig["training"]["batch_size"]

    food_dataset = RFDataset(trainConfig["training"]["pixImgPath"],
                             trainConfig["training"]["trueImgPath"],
                             img_size,
                             NOISE_DIM,
                             transform=T.ToTensor())

    loader_train, loader_val = getTrainAndVal(food_dataset, NUM_TRAIN, batch_size)

    # --- training --- #
    iteration = 0
    for epoch in range(trainConfig["training"]["num_epochs"]):
        for pixImg, trueImg in loader_train:
            if pixImg.shape[0] != 4:
                break
            # --- model update step --- #
            pix2Food.feedInput(pixImg, trueImg)
            pix2Food.optimizeGAN()
            predImg_np = pix2Food.predict()
            # --- show result --- #
            if iteration % trainConfig["training"]["show_every"] == 0:
                print(f"Epoch: {epoch}, Itration: {iteration}")
                trueImg_np = np.copy(trueImg)
                trueImg_np = np.transpose(trueImg_np, (0, 2, 3, 1))
                pix2Food.PrintLossLog()
                show_all_images([pixImg, trueImg_np, predImg_np])
            iteration += 1

    # --- validation --- #
    print("\n\n\n # --- Validation Set --- # \n\n\n")

    for pixImg, trueImg in loader_val:
        if pixImg.shape[0] != 4:
            break
        # --- model update step --- #
        pix2Food.feedInput(pixImg, trueImg)
        predImg_np = pix2Food.predict()
        # --- show result --- #
        trueImg_np = np.copy(trueImg)
        trueImg_np = np.transpose(trueImg_np, (0, 2, 3, 1))
        show_all_images([pixImg, trueImg_np, predImg_np])

    # --- saving model --- #
    modelName = "pix2food.pkl"
    modelPath = os.path.join(opt.checkpoints_dir, modelName)
    with open(modelPath, 'wb') as modeFile:
        pickle.dump(pix2Food, modeFile)
    print(f"save model to {modelPath}")
