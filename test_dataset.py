import yaml
import torchvision.transforms as T
import  numpy as  np

from dataset.mydataset import OriginDataset, getTrainAndVal
from models.utils import show_all_images

if __name__ == "__main__":

    configPath = "dataset.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)

    food_dataset = OriginDataset(trainConfig["data_root"],
                                 trainConfig["img_size"],
                                 transform=T.ToTensor())

    loader_train, loader_val = getTrainAndVal(food_dataset, 100, 4)

    for pixImg, trueImg in loader_train:
        trueImg_np = np.copy(trueImg)
        trueImg_np = np.transpose(trueImg_np, (0, 2, 3, 1))
        show_all_images([pixImg, trueImg_np])
        break
