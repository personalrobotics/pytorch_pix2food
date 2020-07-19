import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images):
    for i in range(len(images)):
        plt.imshow(images[i], interpolation='nearest')
    plt.show()
    return


def show_all_images(imgs_list):
    batch_size = imgs_list[0].shape[0]
    num_row = len(imgs_list)
    # print(batch_size, num_row)
    fig2 = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=batch_size, nrows=num_row, figure=fig2)
    spec2.update(wspace=0.05, hspace=0.05)

    for row, imgs in enumerate(imgs_list):
        for col, img in enumerate(imgs):
            ax = fig2.add_subplot(spec2[row, col])
            ax.axis('off')
            plt.imshow(img, interpolation='nearest')
    # plt.savefig("img.png")
    plt.show()
    return


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0
