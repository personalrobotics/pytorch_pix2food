import torch
import os
import numpy as np
from abc import ABC, abstractmethod

from . import networks
from .utils import preprocess_img, deprocess_img


class Pix2FoodModel(ABC):
    """This defines a general class that can utilize different models to do food picture generation,
    specifically learning a mapping from input images(result from Random Forest) to output images(fake image)
    given paired data.

    """

    def __init__(self, opt, trainConfig):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.trainConfig = trainConfig
        self.batch_size = trainConfig["training"]["batch_size"]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        print(f"self.device is {self.device}")
        self.dtype = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # --- define netG and netD --- #
        self.netG = networks.define_G(netG=opt.netG, gpu_ids=self.gpu_ids).type(self.dtype)

        if self.isTrain:
            self.netD = networks.define_D(netD="patch", gpu_ids=self.gpu_ids).type(self.dtype)
            self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def feedInput(self, pixImg, trueImg):
        self.pixImg = pixImg
        # self.pixVector = self.pixImg.type(self.dtype).view(self.batch_size, -1)
        self.pixVector = self.pixImg.reshape(self.batch_size, -1).type(self.dtype)
        # --- de-mean rgb image --- #
        self.trueImg = preprocess_img(trueImg).to(self.device)

    def backward_G(self):
        self.fakeImg = self.netG(self.pixVector)
        fakeLogits = self.netD(self.fakeImg)
        self.loss_G_GAN = self.criterionGAN(fakeLogits, True)
        self.loss_G_L1 = self.criterionL1(self.fakeImg, self.trueImg)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.opt.lmd
        self.loss_G.backward()

    def backward_D(self):
        # --- D trueImg loss --- #
        trueLogits = self.netD(self.trueImg)
        self.loss_D_true = self.criterionGAN(trueLogits, True)
        # --- D fakeImg loss --- #
        fakeImg = self.netG(self.pixVector).detach()
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
        fakeImg = self.netG(self.pixVector)  # (N, C, H, W)
        fakeImg = deprocess_img(fakeImg)
        fakeImg = fakeImg.data.cpu().numpy()
        fakeImg = np.transpose(fakeImg, (0, 2, 3, 1))  # (N, H, W, C)
        return fakeImg

    def PrintLossLog(self):
        print(f"loss_G_GAN = {self.loss_G_GAN}, loss_G_L1 = {self.loss_G_L1} * {self.opt.lmd}")
        print(f"loss_D = {self.loss_D}")

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
