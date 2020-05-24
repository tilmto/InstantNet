"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


crop_size = 32
padding = 4


def prepare_train_data(dataset='imagenet', datadir='/home/yf22/dataset'):
    if 'imagenet' in dataset:
        train_dataset = torchvision.datasets.ImageFolder(
            datadir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))

    else:
        train_dataset = None
        
    return train_dataset


def prepare_test_data(dataset='imagenet', datadir='/home/yf22/dataset'):

    if 'imagenet' in dataset:
        test_dataset = torchvision.datasets.ImageFolder(datadir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ]))

    else:
        test_dataset = None

    return test_dataset
