import os
import cv2
import numpy as np
from numpy.linalg import norm
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .utils import timeit
# self.tranform = T.Compose([T.ToTensor(),
#                                    T.Resize(size=self.img_size),
#                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


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
        self.down_size = (30, 40)  # (40, 30)
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
        # open the image file and get angle data from the filename
        bbox = self.getBBox(self.imagesName[index].split(".")[0])
        cv2Img = cv2.imread(os.path.join(self.images_path, self.imagesName[index]))
        # --- pixImg --- #
        pixImg = self.generatePixImg(cv2Img, bbox)
        # --- trueImg --- #
        trueImg = Image.open(os.path.join(self.images_path, self.imagesName[index]))
        # --- bboxImg --- #
        bboxImg = np.array(self.getImgWithBBox(trueImg, bbox))
        # --- histImg --- #
        # hist = self.getHistImg(cv2Img, bbox)
        # --- kMeans --- #
        kmPixImg = self.kmeans(cv2.cvtColor(cv2Img, cv2.COLOR_BGR2RGB), bbox)
        # resize and transform the image
        trueImg = trueImg.resize(self.img_size)
        if self.tranform:
            trueImg = self.tranform(trueImg)
        return pixImg, trueImg, bboxImg, kmPixImg

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

    def getHistImg(self, img, bbox):
        xmin, ymin, xmax, ymax = bbox
        hsv = cv2.cvtColor(img[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def getImgWithBBox(self, im, bbox):
        """
            im (PIL.Image)
            bbox (tuple or list)
        """
        # bbox = [50, 50, 70, 70]
        # print(bbox)
        im = im.copy()
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox, width=4)
        return im

    def getBBox(self, imgName):
        tree = ET.parse(os.path.join(self.annotations_path, imgName + ".xml"))
        root = tree.getroot()
        bbox = None
        for node in root:
            if node.tag == 'object':
                obj_name = node.find('name').text
                # print(obj_name)
                if node.find('bndbox') is None:
                    continue
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
        return bbox

    def generatePixImg(self, img, bbox):
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
        # lower_yellow = np.array([18, 20, 180])
        # upper_yellow = np.array([32, 128, 255])
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        res = cv2.bitwise_and(img, img, mask=mask)
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        res = np.where(res > 0, INTENSITY, 0)
        return res


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
