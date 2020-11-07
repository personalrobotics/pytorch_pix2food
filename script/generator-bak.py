import cv2
import os
import torch
import yaml
import rospkg
import rospy
from cv_bridge import CvBridge
from matplotlib import pyplot as plt

from pytorch_pix2food.options.options import BaseOptions
from pytorch_pix2food.models.pix2food_model import Pix2FoodModel
from pytorch_pix2food.dataset.utils import generateActionImg
from pytorch_pix2food.models.utils import show_all_images, show_all_images_rotate, Tensor2Image

from food_msg.srv import Pix2Food, Pix2FoodResponse

rospack = rospkg.RosPack()
pix2food_pkg = rospack.get_path('pytorch_pix2food')
br = CvBridge()

class generator(object):
    
    def __init__(self):
        print("generator init")
        opt = BaseOptions().parse()   # get training options
        base_path = os.path.join(pix2food_pkg, "src/pytorch_pix2food")
        configPath = os.path.join(base_path, "cGAN_config.yaml")
        with open(configPath, 'rb') as file:
            trainConfig = yaml.load(file)
        self.pix2food = Pix2FoodModel(opt, trainConfig)
        ckp_dir = os.path.join(base_path, "checkpoints")
        model_name = "g-patch512.ckp"
        model_path = os.path.join(ckp_dir, model_name)
        self.pix2food.netG.load_state_dict(torch.load(model_path))
        self.service = rospy.Service("pix2food", Pix2Food, self.handle)
        rospy.spin()

    def handle(self, rqt):
        print("process rqt")
        startImg, actImg = rqt.startImg, rqt.actImg
        startImg, actImg = br.imgmsg_to_cv2(startImg), br.imgmsg_to_cv2(actImg)
        cv2.imwrite("startImg.png", startImg)
        cv2.imwrite("actImg.png", actImg)
        fakeImg = self.pred(startImg, actImg)
        fakeImg_msg = br.cv2_to_imgmsg(fakeImg)
        response = Pix2FoodResponse(fakeImg=fakeImg_msg)
        return response

    def generateAction(self, push_direction="left_push", bbox=[105, 218, 425, 262]):
        xmin, ymin, xmax, ymax = bbox
        start = [xmax, (ymin + ymax) // 2]
        end = [xmin, (ymin + ymax) // 2]
        actImg = generateActionImg(start, end, actImg=None, push_direction="left_push", 
                            img_size = (640, 480), forque_width=45)
        return actImg
  
    def pred(self, startImg, actImg):
        """
        Args:
            startImg ([np.ndarray])
            actImg ([np.ndarray])
        """
        print("here")
        self.pix2food.feedNumpyArrayInput(startImg, actImg)
        fakeImg = self.pix2food.predict()
        fakeImg = Tensor2Image(fakeImg).squeeze()
        # fakeImg = cv2.resize(fakeImg, (640, 480))
        cv2.imwrite("fakeImg.png", fakeImg)
        plt.imshow(fakeImg)
        plt.savefig("fake.png")
        return fakeImg

if __name__ == "__main__":
    rospy.init_node("generator")
    g = generator()
    
