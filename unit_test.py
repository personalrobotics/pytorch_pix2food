import yaml
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch

from dataset.mydataset import *
from models.utils import show_all_images, show_images
from models.networks import define_G, ResnetBlock, ResnetDiscriminator


def Tensor2Image(tensor):
    img = np.copy(tensor)
    img = np.transpose(img, (0, 2, 3, 1))
    return img

if __name__ == "__main__":

    configPath = "dataset.yaml"
    with open(configPath, 'rb') as file:
        trainConfig = yaml.load(file, Loader=yaml.FullLoader)
    food_dataset = UNetDataset(trainConfig["data_root"], trainConfig["img_size"], transform=T.ToTensor())
    batch_size = trainConfig["batch_size"]

    # # loader_train, loader_val = getTrainAndVal(food_dataset, 100, batch_size)
    loader_all = DataLoader(food_dataset, batch_size=batch_size)

    # # --- define model --- #
    device = torch.device("cuda:0")
    # # netG = define_G(input_nc=4, output_nc=3, ngf=64, netG="unet_128", norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])
    netG = define_G(input_nc=4, output_nc=3, ngf=64, netG="resnet_6blocks", norm='batch',
                    use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]).to(device)

    print(netG)

    # resnetD = ResnetBlock(64, padding_type="zero", norm_layer=torch.nn.InstanceNorm2d, use_dropout=False, use_bias=False)
    # resnetD = ResnetDiscriminator()
    # resnetD.todevice()

    # print(resnetD)

    # for startImg, actImg, endImg, imgIDs in loader_all:
    #     # print(actImg[0, 0, 50:70, 50:70])
    #     # inputImg = torch.cat((startImg, actImg), 1)
    #     # output = netG(inputImg)
    #     # print(output.shape)
    #     # show_all_images([Tensor2Image(startImg), Tensor2Image(actImg), Tensor2Image(endImg)])
    #     # show_all_images([startImg, actImg, endImg])
    #     print(startImg.shape)
    #     pred = resnetD(startImg)
    #     print(pred.shape)
    #     print(f"imgIDs = {imgIDs}")
    #     # resnetD()
    #     break


