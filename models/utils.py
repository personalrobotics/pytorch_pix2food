import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image

# %matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images):
    for i in range(len(images)):
        plt.imshow(images[i], interpolation='nearest')
    plt.show()
    return


def show_all_images(imgs_list, ratio=(3, 4)):
    batch_size = imgs_list[0].shape[0]
    num_row = len(imgs_list)
    # print(batch_size, num_row)
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=batch_size, nrows=num_row, figure=fig2)
    spec2.update(wspace=0.05, hspace=0.05)

    for row, imgs in enumerate(imgs_list):
        imgs = Tensor2Image(imgs)
        for col, img in enumerate(imgs):
            if img.shape[2] == 1:
                img = img.squeeze()
            if img.shape[0] / img.shape[1] != ratio[0] / ratio[1]:
                img = cv2.resize(img, (640, 480))
            ax = fig2.add_subplot(spec2[row, col])
            ax.axis('off')
            plt.imshow(img, interpolation='nearest')
    # plt.savefig("img.png")
    plt.show()
    return


def show_all_images_rotate(imgs_list, ratio=(3, 4)):
    batch_size = imgs_list[0].shape[0]
    num_col = len(imgs_list)
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=num_col, nrows=batch_size, figure=fig2)
    spec2.update(wspace=0.01, hspace=0.01)

    for col in range(num_col):
        for row in range(batch_size):
            img = imgs_list[col][row]
            img = Tensor2Image(img)
            if img.shape[2] == 1:
                img = img.squeeze()
            if img.shape[0] / img.shape[1] != ratio[0] / ratio[1]:
                img = cv2.resize(img, (640, 480))
            ax = fig2.add_subplot(spec2[row, col])
            ax.axis('off')
            plt.imshow(img, interpolation='nearest')
    plt.show()


def save_images(imgs_list, prefix):
    for row, imgs in enumerate(imgs_list):
        imgs = Tensor2Image(imgs)
        for i, img in enumerate(imgs):
            if img.shape[2] == 1:
                img = img.squeeze()
            # plt.imshow(img, interpolation='nearest')
            new_img = img
            new_img = 255 * new_img
            # new_img = np.zeros((120,160,3), np.uint8)
            # new_img = cv2.normalize(img, new_img, 0, 255, cv2.NORM_MINMAX)

            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            new_img = cv2.resize(new_img, (640, 480))
            cv2.imwrite("rst_imgs/" + prefix + "_" + str(i) + ".png", new_img)
    return


def save_images2(imgs_list, imgIDs):
    for row, imgs in enumerate(imgs_list):
        imgs = Tensor2Image(imgs)
        for i, img in enumerate(imgs):
            if img.shape[2] == 1:
                img = img.squeeze()
            # plt.imshow(img, interpolation='nearest')
            new_img = img
            new_img = 255 * new_img
            # new_img = np.zeros((120,160,3), np.uint8)
            # new_img = cv2.normalize(img, new_img, 0, 255, cv2.NORM_MINMAX)

            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            new_img = cv2.resize(new_img, (640, 480))
            prefix = imgIDs[i]
            cv2.imwrite("rst_imgs/" + prefix + "_5_finish_pred.png", new_img)
    return


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0


def Tensor2Image(tensor):
    # tensor to cv2 image format
    img = np.copy(tensor)
    if len(tensor.shape) == 4:
        img = np.transpose(img, (0, 2, 3, 1))  # (N, H, W, C)
    if len(tensor.shape) == 3:
        img = np.transpose(img,  (1, 2, 0)) # (H, W, C)
    return img
