import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.savefig("img.png")
    # plt.show()
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
            plt.imshow(img)
    # plt.savefig("img.png")
    plt.show()
    return


def preprocess_img(x):
    return 2 * x - 1.0


def deprocess_img(x):
    return (x + 1.0) / 2.0
