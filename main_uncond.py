import copy
import glob
import math
import random
from functools import cmp_to_key

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import Image
from skimage import feature
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange

from CustomDataset import CustomDataset_uncond
from models_uncond import Discriminator, Generator

Tensor = torch.cuda.FloatTensor

gp_factor = 10


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = gp_factor*((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


gp_dra_factor = 10


def compute_gradient_penalty_dra(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = Tensor(np.random.random(size=X.shape))

    interpolates = alpha * X + \
        ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size()).cuda()))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(Tensor(X.shape[0], 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = gp_dra_factor * \
        ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run():
    print('loop')
    # torch.backends.cudnn.enabled = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GaussianNoise.device = device
    # device = torch.device("cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    D = Discriminator().to(device)

    latent_s = 256
    G = Generator(latent_length=latent_s).to(device)

    # G = Generator(1, 100, 3, 64, 0).to(device)
    # D = Discriminator(1, 100, 3, 64, 0).to(device)

    # S = Scaler(3).to(device)

    D.apply(weights_init)
    G.apply(weights_init)
    ld = False
    # gen_name = './gen_ns'
    # fcn_name = './fcn_ns'
    gen_name = './model/gen
    fcn_name = './model/fcn'
    if ld:
        try:
            G.load_state_dict(torch.load(gen_name))
            print('G net loaded')
        except Exception as e:
            print(e)
        try:
            D.load_state_dict(torch.load(fcn_name))
            print('D net loaded')
        except Exception as e:
            print(e)
    # else:
        # D.apply(weights_init_conv)
        # G.apply(weights_init)
        # S.apply(weights_init_conv)

    # face_image_path = '../../datasets/anime-faces/*/*.jpg'
    face_image_path = '../../datasets/portraits/*.jpg'

    plt.ion()

    def cmp(i1):
        i1 = int(i1.replace('\\', '/').split('/')[4].split('.')[0])
        return i1

    # +glob.glob(face_image_path2)
    train_image_paths = glob.glob(face_image_path)
    for i in range(10):
        print(train_image_paths[i])
    print(len(train_image_paths))

    b_size = 32

    train_dataset = CustomDataset_uncond(
        train_image_paths, train=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=b_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=True)

    G.train()
    # for d in D:
    #     d.train()
    D.train()
    # S.train()

    # criterion_CE = nn.CrossEntropyLoss().to(device)
    # critetion = nn.BCEWithLogitsLoss().to(device)
    critetion = nn.BCELoss().to(device)
    criterion_MSE = nn.MSELoss().to(device)

    g_lr = 2e-4
    d_lr = 2e-4
    optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    # optimizer_s = optim.Adam(S.parameters(), lr=g_lr, betas=(0.5, 0.9))

    '''
    g_lr = 5e-5
    d_lr = 5e-5
    optimizer = optim.RMSprop(G.parameters(), lr=g_lr)
    optimizer_d = optim.RMSprop(D.parameters(), lr=d_lr)
    '''
    # optimizer_d = [optim.Adam(D[0].parameters(), lr=d_lr, betas=(0.5, 0.999)), optim.Adam(
    #     D[1].parameters(), lr=d_lr, betas=(0.5, 0.999)), optim.Adam(D[2].parameters(), lr=d_lr, betas=(0.5, 0.999))]

    _zero = torch.from_numpy(
        np.zeros((b_size, 1))).float().to(device)
    _zero.requires_grad = False

    _one = torch.from_numpy(
        np.ones((b_size, 1))).float().to(device)
    _one.requires_grad = False

    # dev = math.exp(-1/math.pi)
    dev = 1

    fixed_z = torch.autograd.Variable(torch.Tensor(
        np.random.normal(0, dev, (64, latent_s, 1, 1)))).to(device)

    for epoch in trange(50, desc='epoch'):
        # print('\nepoch:', epoch)
        '''
        if epoch == 1:
            train_loader.num_workers = 4
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
        '''
        loop = tqdm(train_loader, desc='iteration')
        le = len(train_loader)
        # for batch_idx, (_data, tag) in enumerate(train_loader):
        batch_idx = -1
        routine = 0
        for _data in loop:
            true_prob = 0.7+0.3*(routine/le)
            fake_prob = 1.0-true_prob

            gt = torch.from_numpy((1.0 - true_prob) *
                                  np.random.randn(b_size, 1) + true_prob)
            gt = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy(fake_prob * np.random.randn(b_size, 1))
            fake = fake.type(torch.FloatTensor).to(device)

            # gt = torch.from_numpy(
            #     (1.0 - 0.7) * np.random.randn(b_size, 1) + 0.7)
            # gt = gt.type(torch.FloatTensor).to(device)
            # fake = torch.from_numpy(
            #     (0.3 - 0.0) * np.random.randn(b_size, 1) + 0.0)
            # fake = fake.type(torch.FloatTensor).to(device)
            routine += 1

            batch_idx += 1

            # zero = _zero
            # one = _one

            _data = _data.to(device)
            z = torch.autograd.Variable(torch.Tensor(
                np.random.normal(0, dev, (_data.shape[0], latent_s, 1, 1)))).to(device)
            z.requires_grad = False

            gen, dx = G(z.detach())
            # gen = G(z.detach())

            errD_print = 0

            optimizer_d.zero_grad()
            output2_p = D(_data.detach())
            # errD_true = -torch.mean(output2_p)
            # errD_true = critetion(output2_p, one-0.1)
            errD_true = critetion(output2_p, gt)
            # errD_print += errD_true.item()
            # errD_true.backward()

            output_p = D(gen.detach())
            # errD_gen = torch.mean(output_p)
            errD_gen = critetion(output_p, fake)
            # errD_print += errD_gen.item()
            # errD_gen.backward()

            # Wasserstein GAN-GP
            # gp = compute_gradient_penalty(D, _data.detach(), gen.detach())
            # DRAGAN
            # gp = compute_gradient_penalty_dra(D, _data.detach())
            # gp.backward()
            # errD = -torch.mean(output2_p) + torch.mean(output_p) + gp
            # errD = -torch.mean(output2_p) + torch.mean(output_p)
            # errD = critetion(output2_p, one-0.1) + critetion(output_p, zero)
            errD = 0.5*(errD_gen + errD_true)  # + gp

            errD_print = errD.item()
            errD.backward()
            # errD_print = errD_true.item()+errD_gen.item()
            # ed = (errD_true+errD_gen)/2
            # ed.backward()
            optimizer_d.step()

            # for p in D.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            # if batch_idx % 5 == 0:
            # z = torch.autograd.Variable(torch.Tensor(
            #     np.random.normal(0, dev, (_data.shape[0], latent_s)))).to(device)
            # z.requires_grad = False
            optimizer.zero_grad()
            # gen, dx = G(z.detach())
            output_p = D(gen)

            # if _data.shape[0] % 2 == 0:
            #     im1, im2 = torch.split(gen, _data.shape[0]//2, 0)
            #     r1, r2 = torch.split(z, _data.shape[0]//2, 0)
            # elif _data.shape[0] > 1:
            #     im1, im2 = torch.split(gen[:-1], _data.shape[0]//2, 0)
            #     r1, r2 = torch.split(z[:-1], _data.shape[0]//2, 0)

            # calc_lz = True
            # im_diff = torch.mean(
            #     torch.abs(im1-im2).view(_data.shape[0]//2, -1), axis=1)
            # z_diff = torch.mean(
            #     torch.abs(r1-r2).view(_data.shape[0]//2, -1), axis=1)
            # lz = torch.mean(im_diff / (z_diff))
            # loss_lz = 1/(lz+1e-9)
            # if torch.isnan(loss_lz).any():
            #     calc_lz = False

            # if calc_lz:
            #     loss_lz.backward(retain_graph=True)

            # Wasserstein GAN
            # g_loss = -torch.mean(output_p)
            g_loss = critetion(output_p, gt)
            g_loss.backward()
            # g_loss.backward(retain_graph=True)

            # decode_loss = criterion_MSE(dx, z)
            # decode_loss.backward()

            optimizer.step()

            # loop.set_description(desc="%.4f, %.4f, %.4f" % (
            #     errD_print, g_loss.item(), decode_loss.item()))
            loop.set_description(desc="%.4f, %.4f" %
                                 (errD_print, g_loss.item()))
            # loop.set_description(desc="%.4f, %.4f, %.4f, %.4f" % (
            #     errD_print, g_loss.item(), decode_loss.item(), loss_lz.item()))
            if batch_idx % 10 == 0:

                fig = plt.figure(1, figsize=(8, 8))
                fig.clf()
                gs1 = gridspec.GridSpec(2, 2)
                gs1.update(wspace=0.025, hspace=0.05)

                for i in range(4):
                    ax1 = plt.subplot(gs1[i])
                    plt.axis('off')
                    plt.imshow((np.transpose(
                        gen.detach().cpu().numpy()[i], (1, 2, 0))+1)/2)
                fig.canvas.draw()
                fig.canvas.flush_events()

                # fig = plt.figure(2, figsize=(8, 8))
                # fig.clf()
                # gs2 = gridspec.GridSpec(2, 2)
                # gs2.update(wspace=0.025, hspace=0.05)

                # for i in range(4):
                #     ax1 = plt.subplot(gs2[i])
                #     plt.axis('off')
                #     plt.imshow((np.transpose(
                #         _data.detach().cpu().numpy()[i], (1, 2, 0))+1)/2)
                # fig.canvas.draw()
                # fig.canvas.flush_events()

            if batch_idx % 1000 == 0:
                with torch.no_grad():
                    # G.eval()
                    fake, _ = G(fixed_z.detach())
                    fake = (fake + 1)/2
                    vutils.save_image(fake.data[0:64, :, :, :],
                                      './middle/fake_samples_%03d_%02d.png' % (epoch, batch_idx//100), nrow=8)
                    # G.train()

        torch.save(G.state_dict(), gen_name)
        torch.save(D.state_dict(), fcn_name)
        torch.save(G.state_dict(), f'./{gen_name}_{epoch}')
        torch.save(D.state_dict(), f'./{fcn_name}_{epoch}')

    print('\nFinished Training')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # random.seed()
    # np.random.seed()
    torch.multiprocessing.freeze_support()
    run()
