import os
from dcgan import *
from patchGAN import NLayerDiscriminator, GANLoss, init_weights
# import time
import argparse
import pickle
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)

    return parser.parse_args()


def train(TRAIN=False, SAVE=False, TEST=False):

    NUM_TRAIN = 250
    NOISE_DIM = 1200
    batch_size = 4

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

    # gen_name = "generator-patchGAN-lmd1.pkl"
    # discriminator_name = "discriminator-patchGAN-lmd1.pkl"

    gen_name = "generator-l1loss.pkl"
    discriminator_name = "discriminator-l1loss.pkl"

    if TRAIN:
        # --- discriminator init --- #
        # D_DC = build_dc_classifier().type(dtype)
        # D_DC.apply(initialize_weights)
        # D_DC_solver = get_optimizer(D_DC)
        D_DC = NLayerDiscriminator(input_nc=3).type(dtype)
        init_weights(D_DC)
        D_DC_solver = get_optimizer(D_DC)
        d_loss = GANLoss("lsgan").type(dtype)

        # --- generator init --- #
        G_DC = build_dc_generator(noise_dim=NOISE_DIM).type(dtype)
        G_DC.apply(initialize_weights)
        G_DC_solver = get_optimizer(G_DC)

        # --- run gans --- #
        run_a_gan(loader_train, D_DC, G_DC, D_DC_solver, G_DC_solver, d_loss, generator_loss, show_every=400, batch_size=batch_size, noise_size=NOISE_DIM, num_epochs=200)
    if SAVE:
        pickle.dump(G_DC, open(gen_name, 'wb'))
        pickle.dump(D_DC, open(discriminator_name, 'wb'))
    if TEST:
        G_DC = pickle.load(open(gen_name, 'rb'))
        test_a_gan(loader_test, G_DC, batch_size=batch_size)


if __name__ == "__main__":
    args = parse_args()
    # print(args.train)
    train(TRAIN=args.train, SAVE=args.save, TEST=args.test)
    # print(build_dc_classifier().parameters())
