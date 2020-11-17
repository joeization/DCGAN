import random

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import wiener


class Affine(object):
    '''PyTorch affine adapter

    Args:
        img (PIL image): Images to be affined
    Usage:
        set the args for affine then call Affine(image), see the def for more information
    Returns:
        affined images
    '''
    angle = None
    translations = None
    scale = None
    shear = None

    @staticmethod
    def __call__(img):
        return transforms.functional.affine(img, Affine.angle, Affine.translations, Affine.scale, Affine.shear)


class DatasetStorage():
    storage = {}
    label = {}

    def __init__(self):
        pass


class CsvFetch():
    def __init__(self, csv_path):
        self.hc = {}
        self.ec = {}
        self.label_hair = []
        self.label_eyes = []
        csv = open(csv_path, 'r')
        for g in csv:
            _, color = g.strip().split(',')
            h, _, e, _ = color.split(' ')
            if h not in self.hc:
                self.hc[h] = len(self.hc)
            h = self.hc[h]
            if e not in self.ec:
                self.ec[e] = len(self.ec)
            e = self.ec[e]
            self.label_hair.append(h)
            self.label_eyes.append(e)
        print(f'total {len(self.hc)} hair color(s).')
        print(self.hc)
        print(f'total {len(self.ec)} eye color(s).')
        print(self.ec)

    def __getitem__(self, index):
        return self.label_hair[index], self.label_eyes[index]


class CustomDataset():
    # initial logic happens like transform
    def __init__(self, image_paths, train=True, csv=None):

        # DatasetStorage.storage = {}
        # DatasetStorage.label = {}
        self.image_paths = image_paths
        self.csv = csv
        self.train = train
        self.transforms_distor = transforms.Compose([
            transforms.Grayscale(),
            Affine(),
            transforms.ToTensor(),
        ])
        self.transforms = transforms.Compose([
            # transforms.Grayscale(),
            transforms.Scale(size=(64, 64)),
            # transforms.RandomCrop((32, 32)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        '''
        if index in DatasetStorage.storage:
            return DatasetStorage.storage[index].clone()
            if self.train:
                return DatasetStorage.storage[index].clone()
            else:
                return DatasetStorage.storage[index].clone()
        else:
        '''
        # plt.ion()
        image = Image.open(self.image_paths[index])
        if self.train:
            # Affine.angle, Affine.translations, Affine.scale, Affine.shear = transforms.RandomAffine.get_params(
            #     degrees=(-30, 30), translate=(0.1, 0.1), scale_ranges=(0.95, 1.05), shears=None, img_size=image.size)
            # t_image = self.transforms_distor(image)
            t_image = self.transforms(image)
            # DatasetStorage.storage[index] = t_image.clone()
            # return DatasetStorage.storage[self.image_paths[index]].clone(), DatasetStorage.label[self.image_paths[index]]
            hc, ec = self.csv[index]
            return t_image, hc, ec

        else:
            t_image = self.transforms(image)
            # DatasetStorage.storage[index] = t_image.clone()
            # return DatasetStorage.storage[self.image_paths[index]].clone()
            return t_image

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)


class CustomDataset_uncond():
    # initial logic happens like transform
    def __init__(self, image_paths, train=True, csv=None):

        # DatasetStorage.storage = {}
        # DatasetStorage.label = {}
        self.image_paths = image_paths
        self.csv = csv
        self.train = train
        self.transforms_distor = transforms.Compose([
            transforms.Grayscale(),
            Affine(),
            transforms.ToTensor(),
        ])
        image_size = 128
        self.transforms = transforms.Compose([
            transforms.Resize(int(image_size*1.3)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        '''
        if index in DatasetStorage.storage:
            return DatasetStorage.storage[index].clone()
            if self.train:
                return DatasetStorage.storage[index].clone()
            else:
                return DatasetStorage.storage[index].clone()
        else:
        '''
        # plt.ion()
        image = Image.open(self.image_paths[index])
        if self.train:
            # Affine.angle, Affine.translations, Affine.scale, Affine.shear = transforms.RandomAffine.get_params(
            #     degrees=(-30, 30), translate=(0.1, 0.1), scale_ranges=(0.95, 1.05), shears=None, img_size=image.size)
            # t_image = self.transforms_distor(image)
            t_image = self.transforms(image)
            # DatasetStorage.storage[index] = t_image.clone()
            # return DatasetStorage.storage[self.image_paths[index]].clone(), DatasetStorage.label[self.image_paths[index]]
            return t_image

        else:
            t_image = self.transforms(image)
            # DatasetStorage.storage[index] = t_image.clone()
            # return DatasetStorage.storage[self.image_paths[index]].clone()
            return t_image

    def __len__(self):  # return count of sample we have

        return len(self.image_paths)
