import cv2
import os
import torch
from PIL import Image
import pickle
import yaml
import rospkg
import rospy
from cv_bridge import CvBridge
from matplotlib import pyplot as plt
from torchvision import transforms as T

from pytorch_pix2food.options.options import BaseOptions
from pytorch_pix2food.models.pix2food_model import Pix2FoodModel
from pytorch_pix2food.dataset.utils import generateActionImg
from pytorch_pix2food.models.utils import Tensor2Image

from food_msg.srv import Pix2Food, Pix2FoodResponse

rospack = rospkg.RosPack()
pix2food_pkg = rospack.get_path('pytorch_pix2food')
br = CvBridge()

class Generator(object):
    
    def __init__(self):
        rospy.init_node("generator")
        self._model_init()
        self._service_init()
        rospy.spin()

    def _model_init(self):
        opt = BaseOptions().parse()   # get training options
        base_path = os.path.join(pix2food_pkg, "src/pytorch_pix2food")
        configPath = os.path.join(base_path, "cGAN_config.yaml")
        with open(configPath, 'rb') as file:
            trainConfig = yaml.load(file)
        self.pix2food = Pix2FoodModel(opt, trainConfig)
        ckp_dir = os.path.join(base_path, "checkpoints")
        # model_name = "pix2food-patch512.pkl"
        model_name = "py2-pix2food-patch512.pkl"
        model_path = os.path.join(ckp_dir, model_name)
        with open(model_path, 'rb') as modelFile:
            self.pix2food = pickle.load(modelFile)
        self.tranform = T.Compose([T.Resize(size=(512,512)),
                                   T.ToTensor()])
        rospy.loginfo("finish model init")

    def _service_init(self):
        rospy.Service("pix2food", Pix2Food, self._handle)
        rospy.loginfo("finish service init")

    def _handle(self, rqt):
        rospy.loginfo("process one request")
        startImg, actImg = br.imgmsg_to_cv2(rqt.startImg), br.imgmsg_to_cv2(rqt.actImg)
        startImg, actImg = Image.fromarray(startImg), Image.fromarray(actImg)
        if self.tranform:
            startImg = torch.unsqueeze(self.tranform(startImg), 0)
            actImg = torch.unsqueeze(self.tranform(actImg), 0)
        pixImg = torch.cat((startImg, actImg), 1)
        # print(pixImg.shape)
        self.pix2food.feedInput(pixImg)
        fakeImg = self.pix2food.predict()
        fakeImg = Tensor2Image(fakeImg[0])
        fakeImg = cv2.resize(fakeImg, (640, 480))
        fakeImg = br.cv2_to_imgmsg(fakeImg)
        return Pix2FoodResponse(fakeImg=fakeImg)

    def getTestInputImg(self):
        # -------------- Action IMG --------------- #
        bbox=[75, 118, 300, 162]
        xmin, ymin, xmax, ymax = bbox
        start = [xmax, (ymin + ymax) // 2]
        end = [xmin, (ymin + ymax) // 2]
        actImg = generateActionImg(start, end, actImg=None, push_direction="left_push", 
                                img_size = (640, 480), forque_width=45)
        actImg = Image.fromarray(actImg)
        # -------------- Start IMG --------------- #
        script_base = os.path.join(pix2food_pkg, "script")
        img_path = os.path.join(script_base, "0060_1_start.png")
        startImg = Image.open(img_path)
        if self.tranform:
            self.startImg = torch.unsqueeze(self.tranform(startImg), 0)
            self.actImg = torch.unsqueeze(self.tranform(actImg), 0)

    def test(self):
        pixImg = torch.cat((self.startImg, self.actImg), 1)
        self.pix2food.feedInput(pixImg)
        fakeImg = self.pix2food.predict()
        fakeImg = Tensor2Image(fakeImg[0])
        fakeImg = cv2.resize(fakeImg, (640, 480))
        plt.imshow(fakeImg)
        plt.show()

if __name__ == "__main__":
    g = Generator()

