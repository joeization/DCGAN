import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()
        self.base = base
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, padding=1, bias=False),
            # nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base*2, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*2, base*4, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*4, base*8, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(base*8, 1, 4, 1, padding=0, bias=False),
            # nn.Sigmoid(),

            nn.Conv2d(base*8, base*16, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(base*16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*16, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1)
        return x


class Generator(nn.Module):
    def __init__(self, latent_length=64):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.latent = latent_length

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.latent, 64*16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*16),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*16, 64*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        inp = x.view(-1, self.latent, 1, 1)
        ret = self.conv(inp)
        return ret, 0


class Discriminator_no_stride(nn.Module):
    def __init__(self, base=64):
        super(Discriminator_no_stride, self).__init__()
        self.base = base
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(base, base*2, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(base*2, base*4, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(base*4, base*8, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(base*8, 1, 3, 1, padding=1, bias=False),
            # nn.Sigmoid(),

            nn.MaxPool2d(2, 2),
            nn.Conv2d(base*8, base*16, 5, 1, padding=2, bias=False),
            nn.BatchNorm2d(base*16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base*16, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1)
        return x


class Generator_no_stride(nn.Module):
    def __init__(self, latent_length=64):
        super(Generator_no_stride, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.latent = latent_length

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.latent, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64*8, 64*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64*4, 64*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64*2, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh(),
        )


    def forward(self, x):
        inp = x.view(-1, self.latent, 1, 1)
        ret = self.conv(inp)
        return ret, 0


class Discriminator_fc(nn.Module):
    def __init__(self, output_dim=23, hc=12, ec=10, base=64):
        super(Discriminator_fc, self).__init__()
        self.output_dim = output_dim
        self.base = base
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base, base*2, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base*2, base*4, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base*4, base*8, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(base*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base*8, base*16, 4, 1, padding=0, bias=False),
            nn.BatchNorm2d(base*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_t = nn.Sequential(
            nn.Linear(base*16, base*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base*16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.base*16)
        t = self.fc_t(x)
        return t


class Generator_fc(nn.Module):
    def __init__(self, latent_length=64, middle_length=1024):
        super(Generator_fc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.m = middle_length
        self.latent_map = nn.Sequential(
            nn.Linear(latent_length, middle_length),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(middle_length, middle_length),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
        )
        self.pred = nn.Sequential(
            nn.Linear(middle_length, middle_length),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(middle_length, latent_length),
        )

        self.activate = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(middle_length, 64*8, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*8)
        self.conv2 = nn.Conv2d(64*8, 64*4, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.conv3 = nn.Conv2d(64*4, 64*2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.conv4 = nn.Conv2d(64*2, 64, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.latent_map(x)
        decode_x = self.pred(x)
        inp = x.view(-1, self.m, 1, 1)
        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv1(inp)
        inp = self.bn1(inp)
        inp = self.activate(inp)

        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv2(inp)
        inp = self.bn2(inp)
        inp = self.activate(inp)

        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv3(inp)
        inp = self.bn3(inp)
        inp = self.activate(inp)

        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv4(inp)
        inp = self.bn4(inp)
        inp = self.activate(inp)

        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv5(inp)
        inp = self.bn5(inp)
        inp = self.activate(inp)

        inp = F.interpolate(inp, scale_factor=2,
                            mode='bilinear', align_corners=True)
        inp = self.conv6(inp)
        inp = torch.tanh(inp)
        return inp, decode_x
