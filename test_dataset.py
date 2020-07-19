import yaml
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from time import time

from dataset.mydataset import OriginDataset, getTrainAndVal
from models.utils import show_all_images, show_images

if __name__ == "__main__":

    configPath = "dataset.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)

    food_dataset = OriginDataset(trainConfig["data_root"],
                                 trainConfig["img_size"],
                                 transform=T.ToTensor())
    batch_size = trainConfig["batch_size"]

    loader_train, loader_val = getTrainAndVal(food_dataset, 100, batch_size)
    loader_all = DataLoader(food_dataset, batch_size=batch_size)

    for pixImg, trueImg, bboxImg, kMeansImg in loader_train:
        trueImg_np = np.copy(trueImg)
        trueImg_np = np.transpose(trueImg_np, (0, 2, 3, 1))
        show_all_images([pixImg, trueImg_np, bboxImg, kMeansImg])
        break
