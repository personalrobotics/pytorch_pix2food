import os
import cv2
from PIL import Image

from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from torch.utils.data import DataLoader

import xml.etree.ElementTree as ET
import numpy as np

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class OriginDataset(Dataset):

    def __init__(self, data_root, img_size, transform=None):
        self.down_size = (30, 40)  # (40, 30)
        self.data_root = data_root
        self.tranform = transform
        self.img_size = img_size
        self.images_path = os.path.join(self.data_root, "images")
        self.annotations_path = os.path.join(self.data_root, "annotations")
        self.imagesName = os.listdir(self.images_path)

    def __getitem__(self, index):
        # open the image file and get angle data from the filename
        pixImg = cv2.imread(os.path.join(self.images_path, self.imagesName[index]))
        pixImg = self.generatePixImg(pixImg, self.imagesName[index].split(".")[0])

        trueImg = Image.open(os.path.join(self.images_path, self.imagesName[index]))
        trueImg = trueImg.resize((self.img_size[0], self.img_size[1]))
        # transform the image
        if self.tranform:
            trueImg = self.tranform(trueImg)
        return pixImg, trueImg

    def generatePixImg(self, img, imgName):
        tree = ET.parse(os.path.join(self.annotations_path, imgName + ".xml"))
        root = tree.getroot()
        bbox = None
        for node in root:
            if node.tag == 'object':
                obj_name = node.find('name').text
                print(obj_name)
                if node.find('bndbox') is None:
                    continue
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
        img = self.crop(img, bbox)
        img = self.binarize(img)
        return img

    def crop(self, img, bbox):
        xmin, ymin, xmax, ymax = bbox
        # x, y = xmin, ymin
        # w, h = ymax - ymin, xmax - xmin
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # bai_bb = np.copy(img)
        bai_clean = np.copy(img)

        # cv2.rectangle(bai_bb, (x, y), (x+h, y+w), (255, 255, 255), 2)

        mask = np.zeros_like(img)
        mask[ymin:ymax, xmin:xmax] = 1.0
        bai_clean = mask * img
        return bai_clean

    def binarize(self, img):
        # Convert BGR to HSV
        INTENSITY = 1

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # define range of yellow color in HSV
        lower_yellow = np.array([18, 20, 180])
        upper_yellow = np.array([32, 128, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        res = cv2.bitwise_and(img, img, mask=mask)
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res = np.where(res > 0, INTENSITY, 0)
        return res

    def __len__(self):
        #return the total number of dataset
        return len(self.imagesName)


class RFDataset(Dataset):
    """[summary]

    Args:
        input_path (str): path to pix image
        image_path (str): path to ground truth image
    """
    def __init__(self, input_path, image_path, img_size, noise_dim=1200, transform=T.ToTensor):
        self.input_path = input_path
        self.image_path = image_path
        self.tranform = transform
        self.img_size = img_size
        self.input_paths = os.listdir(self.input_path)
        # print(self.input_paths)

    def __getitem__(self, index):
        # # open the image file and get angle data from the filename
        input_img = cv2.imread(os.path.join(self.input_path, self.input_paths[index]), 0)

        image = Image.open(os.path.join(self.image_path, self.input_paths[index]))
        image = image.resize((self.img_size[0], self.img_size[1]))

        # transform the image
        if self.tranform:
            image = self.tranform(image)
        return input_img, image
        # return image

    def __len__(self):
        #return the total number of dataset
        return len(self.input_paths)


def getTrainAndVal(dataset, NUM_TRAIN, batch_size):
    NUM_ALL = len(dataset)
    if NUM_TRAIN > NUM_ALL:
        raise ValueError("# train > # the whole dataset")
    lengths = [NUM_TRAIN, NUM_ALL - NUM_TRAIN]
    train_data, val_data = random_split(dataset, lengths)
    loader_train = DataLoader(train_data, batch_size=batch_size)
    loader_val = DataLoader(val_data, batch_size=batch_size)
    return loader_train, loader_val
