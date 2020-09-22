import os
import cv2
import numpy as np
from numpy.linalg import norm
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .utils import timeit, kmeans
# self.tranform = T.Compose([T.ToTensor(),
#                            T.Resize(size=self.img_size),
#                            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def getTrainAndVal(dataset, NUM_TRAIN, batch_size):
    NUM_ALL = len(dataset)
    if NUM_TRAIN > NUM_ALL:
        raise ValueError("# train > # the whole dataset")
    lengths = [NUM_TRAIN, NUM_ALL - NUM_TRAIN]
    train_data, val_data = random_split(dataset, lengths)
    loader_train = DataLoader(train_data, batch_size=batch_size)
    loader_val = DataLoader(val_data, batch_size=batch_size)
    return loader_train, loader_val


class OriginDataset(Dataset):

    def __init__(self, data_root, img_size, transform=None):
        self.down_size = (40, 30)  # (40, 30)

        self.data_root = data_root
        self.img_size = img_size
        self.images_path = os.path.join(self.data_root, "images")
        self.annotations_path = os.path.join(self.data_root, "annotations")
        self.imagesName = os.listdir(self.images_path)
        # --- transform --- #
        self.tranform = transform

    def __len__(self):
        #return the total number of dataset
        return len(self.imagesName)

    def __getitem__(self, index):
        INTENSITY = 1
        # open the image file and get angle data from the filename
        bbox = self.getBBox(self.imagesName[index].split(".")[0])
        cv2Img = cv2.imread(os.path.join(self.images_path, self.imagesName[index]))
        # --- kMeans --- #
        kmPixImg = self.kmeans(cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB), bbox)
        kmPixImg = cv2.cvtColor(kmPixImg, cv2.COLOR_RGB2GRAY)
        kmPixImg = cv2.resize(kmPixImg, self.down_size)
        kmPixImg = np.where(kmPixImg > 0, INTENSITY, 0)
        # --- trueImg --- #
        trueImg = Image.open(os.path.join(self.images_path, self.imagesName[index]))
        # --- bboxImg --- #
        bboxImg = np.array(self.getImgWithBBox(trueImg, bbox))
        # resize and transform the image
        trueImg = trueImg.resize(self.img_size)
        if self.tranform:
            trueImg = self.tranform(trueImg)
        return kmPixImg, trueImg, bboxImg

    def getBBox(self, imgName):
        tree = ET.parse(os.path.join(self.annotations_path, imgName + ".xml"))
        root = tree.getroot()
        bbox = None
        for node in root:
            if node.tag == 'object':
                # obj_name = node.find('name').text
                # print(obj_name)
                if node.find('bndbox') is None:
                    continue
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
        return bbox

    def getImgWithBBox(self, im, bbox):
        """
            im (PIL.Image)
        """
        im = im.copy()
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox, width=4)
        return im

    # @timeit
    def kmeans(self, img, bbox):
        xmin, ymin, xmax, ymax = bbox
        roi = img[ymin:ymax, xmin:xmax]
        Z = roi.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        # print(f"label = {label.shape}, center = {center.shape}")
        center = np.uint8(center)
        blue = np.array([23, 66, 109])
        light = np.array([255] * 3)
        dark = np.array([0] * 3)
        if norm(center[0] - blue) < norm(center[1] - blue):
            center = np.array([dark, light])
        else:
            center = np.array([light, dark])
        res = center[label.flatten()].reshape((roi.shape))
        kmPixImg = np.zeros_like(img)
        kmPixImg[ymin:ymax, xmin:xmax] = res
        return kmPixImg


class UNetDataset(OriginDataset):

    def __init__(self, data_root, img_size, transform=None, test=False):
        OriginDataset.__init__(self, data_root, img_size, transform=transform)
        self.forque_width = 45  # in pixel after down sample
        self.origin_size = (640, 480)
        self.start_path = os.path.join(self.data_root, "start")
        self.end_path = os.path.join(self.data_root, "end")
        if test:
            self.start_path = os.path.join(self.data_root, "test_start")
            self.end_path = os.path.join(self.data_root, "test_end")
        self.startName = os.listdir(self.start_path)

    def __len__(self):
        return len(self.startName)

    def collate_fn(self, batch):
        startImg = [x[0] for x in batch]
        actImg = [x[1] for x in batch]
        endImg = [x[2] for x in batch]
        imgID = [x[3] for x in batch]
        plateBBox = [x[4] for x in batch]

        startImg = torch.stack(startImg)
        actImg = torch.stack(actImg)
        endImg = torch.stack(endImg)
        # imgID = torch.stack(imgID)
        # plateBBox = plateBBox
        return startImg, actImg, endImg, imgID, plateBBox

    def __getitem__(self, index):
        # open the image file and get angle data from the filename
        imgID = self.startName[index].split(".")[0][:4]
        # print(f"imgID = {imgID}")
        startName = imgID + "_1_start.png"
        endName = imgID + "_5_finish.png"
        # --- prepare data --- #
        startImg = Image.open(os.path.join(self.start_path, startName))
        endImg = Image.open(os.path.join(self.end_path, endName))

        actImg = cv2.imread(os.path.join(self.start_path, startName))
        actImg = cv2.cvtColor(actImg, cv2.COLOR_BGR2RGB)
        # actImg = self.getActionImgFromXML(endName, actImg=np.array(startImg))
        actImg = self.getActionImgFromXML(endName.split(".")[0], actImg=None)
        actImg = Image.fromarray(actImg)
        # resize and transform the image
        startImg = startImg.resize(self.img_size)
        endImg = endImg.resize(self.img_size)
        actImg = actImg.resize(self.img_size)
        if self.tranform:
            startImg = self.tranform(startImg)
            endImg = self.tranform(endImg)
            actImg = self.tranform(actImg)

        plateBBox = self.getBBox(endName.split(".")[0], ojbName="plate")

        return startImg, actImg, endImg, imgID, plateBBox

    def getBBox(self, imgName, ojbName):
        tree = ET.parse(os.path.join(self.annotations_path, imgName + ".xml"))
        root = tree.getroot()
        bbox = None
        for node in root:
            if node.tag == 'object' and node.find("name").text == ojbName:
                if node.find('bndbox') is None:
                    continue
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
                bbox = (bbox)
        return bbox

    def getActionImgFromXML(self, endName, actImg=None):
        INTENSITY = 255
        if actImg is None:
            actImg = np.zeros((self.origin_size[1], self.origin_size[0]))
        tree = ET.parse(os.path.join(self.annotations_path, endName + ".xml"))
        root = tree.getroot()
        bbox = None
        ojbName = "left_push"
        # print(endName)
        for node in root:
            if node.tag == 'object' and node.find("name").text == ojbName:
                if node.find("bndbox") is None:
                    continue
                xmin = int(node.find("bndbox").find('xmin').text)
                ymin = int(node.find("bndbox").find('ymin').text)
                xmax = int(node.find("bndbox").find('xmax').text)
                ymax = int(node.find("bndbox").find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
        # --- assume it's left  push --- #
        start = [xmax, (ymin + ymax) // 2]
        end = [xmin, (ymin + ymax) // 2]
        # TODO: cover cases where direction of push action are arbitrary.
        # Here we only cover `left` direction.
        distance_in_pixel = start[0] - end[0]
        x, y = start
        r = self.forque_width // 2
        for i in range(distance_in_pixel):
            actionValue = INTENSITY * i / distance_in_pixel
            actImg[y - r: y + r, x - i] = actionValue
        cv2.normalize(actImg, actImg, 0, 255, cv2.NORM_MINMAX)
        return actImg
