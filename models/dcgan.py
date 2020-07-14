from mydataset import *
from utils import *
import functools

import torch.nn as nn
import torch

import numpy as np


# Deeply Convolutional GANs
# checkParam
def ConvParams(input_size, kernel_size, stride=1, padding=1):
    """
        input_size: np.ndarray
    """
    data = (input_size + 2 * padding - kernel_size) / stride
    # assert isinstance(data, int)
    # print(data)
    output_size = np.floor(data).astype(int) + 1

    return output_size


def getD_ConvTransposeParams(img_size):
    img_size = np.array(img_size)
    assert len(img_size) == 2
    size = ConvParams(img_size, kernel_size=5, stride=1, padding=0)
    # print(size)
    size = ConvParams(size, kernel_size=2, stride=2, padding=0)
    # print(size)
    size = ConvParams(size, kernel_size=5, stride=1, padding=0)
    # print(size)
    size = ConvParams(size, kernel_size=2, stride=2, padding=0)
    return size


D_output_size = getD_ConvTransposeParams(img_size)
w, h = D_output_size
# print(D_output_size)


def build_dc_classifier():
    """
    Build and return a PyTorch model for the DCGAN discriminator implementing
    the architecture above.
    """
    return nn.Sequential(
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        Unflatten(-1, 3, img_size[1], img_size[0]),
        # Unflatten(-1, 3, patch_size[1], patch_size[0]),
        nn.Conv2d(3, 32, (5,5), stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d((2,2), stride=2),

        nn.Conv2d(32, 64, (5,5), stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d((2,2), stride=2),

        Flatten(),

        nn.Linear(64 * w * h, 64 * 4 * 4),
        nn.LeakyReLU(0.01),

        nn.Linear(4 * 4 * 64, 1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )


def GParams(input_size, kernel_size, stride=1, padding=1):
    """
        input_size: np.ndarray
    """
    data = (input_size + 2 * padding - kernel_size) / stride
    # assert isinstance(data, int)
    # print(data)
    output_size = np.floor(data).astype(int) + 1
    return output_size


def getG_ConvTransposeParams(img_size):
    img_size = np.array(img_size)
    assert len(img_size) == 2
    size = GParams(img_size, kernel_size=4, stride=2, padding=1)
    size = GParams(size, kernel_size=4, stride=2, padding=1)
    return size


output_size = getG_ConvTransposeParams(img_size)
Gw, Gh = output_size
# print(output_size)


def build_dc_generator(noise_dim=1024):
    """
    Build and return a PyTorch model implementing the DCGAN generator using
    the architecture described above.
    """
    return nn.Sequential(
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),

        nn.Linear(1024, 128 * Gw * Gh),
        nn.ReLU(),
        nn.BatchNorm1d(128 * Gw * Gh),

        Unflatten(-1, 128, Gh, Gw),

        nn.ConvTranspose2d(128, 64, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),

        nn.ConvTranspose2d(64, 3, (4,4), stride=2, padding=1),
        nn.Tanh(),

        Flatten()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    )

# train a gan
def run_a_gan(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=700,
              batch_size=4, noise_size=96, num_epochs=100):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    # D_losses = []
    # G_losses = []
    for epoch in range(num_epochs):
        for input_vector, x in loader_train:
            input_img = np.copy(input_vector)
            true_img = np.copy(x)
            input_vector = input_vector.reshape(batch_size, -1)
            if len(x) != batch_size:
                # print("last incomplete batch in {}th epoch".format(epoch))
                continue

            # --- discriminator loss --- #
            D_solver.zero_grad()
            real_data = x.type(dtype)
            logits_real = D(2 * (real_data - 0.5)).type(dtype)

            # g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            g_fake_seed = input_vector.type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 3, img_size[1], img_size[0]))

            # d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error = discriminator_loss(logits_real, True) + discriminator_loss(logits_fake, False)
            d_total_error.backward()
            D_solver.step()

            # --- generator loss --- #
            G_solver.zero_grad()
            # g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 3, img_size[1], img_size[0]))
            g_error = 0
            g_error = generator_loss(gen_logits_fake)

            # ****** add L1 loss ****** #
            fake_torch = fake_images.view(batch_size, 3, img_size[1], img_size[0])
            L1_loss = torch.mean(torch.abs(fake_torch - 2 * (real_data - 0.5)))
            lmd = 2.0
            # cg_error = g_error + lmd * L1_loss
            cg_error = L1_loss

            cg_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):

                print('Epoch:{}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_total_error.item(),g_error.item()))
                print(f"gans_loss{g_error}, L1_loss{L1_loss}")
                fake_images = deprocess_img(fake_images)
                # print("fake from G, size = ", fake_images.shape)
                imgs = fake_images.view(batch_size, 3, img_size[1], img_size[0])
                imgs = imgs.data.cpu().numpy()
                imgs = np.transpose(imgs, (0, 2, 3, 1))

                true_img = np.transpose(true_img, (0, 2, 3, 1))
                show_all_images([input_img, imgs, true_img])
                plt.show()
                print("------------------------------")
            iter_count += 1
            # print("iter = ", iter_count)


def test_a_gan(loader_test, G, batch_size=4):
    for input_vector, true_img in loader_test:
        if len(input_vector) != batch_size:
            continue
        input_img = np.copy(input_vector)
        input_vector = input_vector.reshape(batch_size, -1).type(dtype)
        fake_images = G(input_vector)
        # print(fake_images.shape)
        fake_images = deprocess_img(fake_images)
        fake_images = fake_images.view(batch_size, 3, img_size[1], img_size[0])
        imgs = fake_images.data.cpu().numpy()
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        true_img = np.transpose(true_img, (0, 2, 3, 1))
        show_all_images([input_img, imgs, true_img])

