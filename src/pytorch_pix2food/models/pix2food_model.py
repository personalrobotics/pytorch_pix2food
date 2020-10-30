import torch
from torchvision import transforms as T
import os
import numpy as np
from abc import abstractmethod

from . import networks
from .utils import preprocess_img, deprocess_img
from PIL import Image


class Pix2FoodModel(object):
    """This defines a general class that can utilize different models to do food picture generation,
    specifically learning a mapping from input images(result from Random Forest) to output images(fake image)
    given paired data.

    """

    def __init__(self, opt, config):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.img_size = config["training"]["img_size"]
        self.config = config
        self.lmd = self.config["gen"]["lmd"]
        self.batch_size = config["training"]["batch_size"]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # print(f"self.device is {self.device}")
        self.dtype = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # transform
        self.tranform = T.Compose([T.Resize(size=self.img_size),
                                   T.ToTensor()])

        # --- define netG and netD --- #
        self.netG = networks.define_G(input_nc=config["gen"]["input_nc"],netG=config["gen"]["netG"], gpu_ids=self.gpu_ids).type(self.dtype)

        if self.isTrain:
            self.netD = networks.define_D(netD="patch", gpu_ids=self.gpu_ids).type(self.dtype)
            print(self.netD)
            # print(f"trainable params = {networks.count_parameters(self.netD)}")
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def feedNumpyArrayInput(self, startImg, actImg, trueImg=None):
        startImg, actImg = Image.fromarray(startImg), Image.fromarray(actImg)
        startImg, actImg = self.tranform(startImg), self.tranform(actImg)
        if len(startImg.shape) < 4 or len(actImg.shape) < 4:
            startImg = torch.unsqueeze(startImg, 0)
            actImg = torch.unsqueeze(actImg, 0)
        pixImg = torch.cat((startImg, actImg), 1)
        self.pixImg = preprocess_img(pixImg).to(self.device)

    def feedInput(self, pixImg, trueImg=None):
        self.pixImg = preprocess_img(pixImg).to(self.device)
        # self.pixVector = self.pixImg.type(self.dtype).view(self.batch_size, -1)
        # self.pixVector = self.pixImg.reshape(self.batch_size, -1).type(self.dtype)
        # --- de-mean rgb image --- #
        if trueImg is not None:
            self.trueImg = preprocess_img(trueImg).to(self.device)

    def backward_G(self):
        self.fakeImg = self.netG(self.pixImg)
        fakeLogits = self.netD(self.fakeImg)
        self.loss_G_GAN = self.criterionGAN(fakeLogits, True)
        self.loss_G_L1 = self.criterionL1(self.fakeImg, self.trueImg)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lmd
        self.loss_G.backward()

    def backward_D(self):
        # --- D trueImg loss --- #
        trueLogits = self.netD(self.trueImg)
        self.loss_D_true = self.criterionGAN(trueLogits, True)
        # --- D fakeImg loss --- #
        fakeImg = self.netG(self.pixImg).detach()
        fakeLogits = self.netD(fakeImg)
        self.loss_D_fake = self.criterionGAN(fakeLogits, False)
        # --- D total loss --- #
        self.loss_D = (self.loss_D_true + self.loss_D_fake) * 0.5
        self.loss_D.backward()

    def optimizeGAN(self):
        # --- D update --- #
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # --- G update --- #
        # self.set_requires_grad(self.netG, True)  # enable backprop for D
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def predict(self):
        fakeImg = self.netG(self.pixImg)  # (N, C, H, W)
        fakeImg = deprocess_img(fakeImg)
        fakeImg = fakeImg.data.cpu().numpy()
        return fakeImg

    def PrintLossLog(self):
        # print(f"loss_G_GAN = {self.loss_G_GAN}, loss_G_L1 = {self.loss_G_L1} * {self.lmd}")
        # print(f"loss_D = {self.loss_D}")
        pass

    def setlmd(self, lmd):
        self.lmd = lmd

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
