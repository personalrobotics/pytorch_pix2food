from patchGAN import *
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np

patchDiscriminator = NLayerDiscriminator(input_nc=3)
init_weights(patchDiscriminator)
# patchDiscriminator.type(dtype)
patchDiscriminator.to(0)
# print(patchDiscriminator)

NUM_TRAIN = 250
NOISE_DIM = 1200
batch_size = 1

transform = T.Compose([
    T.ToTensor(),
    # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# home = os.path.expanduser("~")
data1 = "/home/nansong/Dropbox/collaborative_ws/net_ws/src/randomforest/data/images/origin"
data2 = "/home/nansong/Dropbox/collaborative_ws/net_ws/src/randomforest/data/images/final"
# food_train = MyDataset(data1, img_size, transform)
food_train = RFDataset(data2, data1, img_size, NOISE_DIM, transform)
all_length = len(food_train)
NUM_VAL = all_length - NUM_TRAIN
lengths = [NUM_TRAIN, all_length - NUM_TRAIN]
train_data, val_data = torch.utils.data.random_split(food_train, lengths)
loader_train = DataLoader(train_data, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0))
loader_test = DataLoader(val_data, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, 0))

for input_vector, x in loader_train:
    # --- show img --- #
    true_img = np.copy(x)
    true_img = np.transpose(true_img, (0, 2, 3, 1))
    show_all_images([true_img])
    # --- discriminator forward --- #
    x = x.type(dtype)
    rst = patchDiscriminator(x)
    print(patchDiscriminator.parameters())
    print(rst.shape)
    # --- gan loss ---#
    criterionGAN = GANLoss("vanilla").to(0)
    print(criterionGAN)
    print(criterionGAN(rst, False))
    break
    # result =
