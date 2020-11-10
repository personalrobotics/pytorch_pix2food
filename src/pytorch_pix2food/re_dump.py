import cv2
import os
import torch
import pickle

from pytorch_pix2food.models.pix2food_model import Pix2FoodModel

class generator(object):
    def __init__(self):
        print("init")
        self.pix2food = None
        self.model_py2 = None
        self.init_model()
        self.save_state_dict()
        # self.load_python2()

    def init_model(self):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        # device = torch.device("cpu") 
        ckp_dir = "/home/nansong/Dropbox/collaborative_ws/my_ws/src/pytorch_pix2food/src/pytorch_pix2food/checkpoints"
        name = "pix2food-patch512.pkl"
        model_path = os.path.join(ckp_dir, name)
        with open(model_path, "rb") as modelFile:
            self.pix2food = pickle.load(modelFile)
        # self.pix2food.netG.to(device)
        # print(next(self.pix2food.netG.parameters()).device)
        # torch.save(self.pix2food.netG.state_dict(), "checkpoints/g-patch512.ckp", pickle_protocol=2)
        # with open("checkpoints/pix2food-new512.ckp", "wb") as f:
        #     pickle.dump(self.pix2food, f, protocol=2)

    def save_state_dict(self):
        model = self.pix2food.netG
        netG_path = "checkpoints/netG-512.pt"
        torch.save(model.state_dict(), netG_path, pickle_protocol=2)

    def load_python2(self):
        ckp_dir = "/home/nansong/Dropbox/collaborative_ws/my_ws/src/pytorch_pix2food/src/pytorch_pix2food/checkpoints"
        name = "pix2food-new512.ckp"        
        model_path = os.path.join(ckp_dir, name)
        with open(model_path, "rb") as modelFile:
            self.model_py2 = pickle.load(modelFile)
        torch.save(self.model_py2.netG.state_dict(), "checkpoints/g-patch512.ckp", pickle_protocol=2)
if __name__ == "__main__":
    g = generator()
