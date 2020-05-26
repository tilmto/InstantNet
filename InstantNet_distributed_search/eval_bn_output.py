from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config

from model_search import FBNet as Network
from model_infer import FBNet_Infer

from thop import profile
from thop.count_hooks import count_convNd

import genotypes

import operations
from quantize import QConv2d

from calibrate_bn import bn_update

operations.DWS_CHWISE_QUANT = config.dws_chwise_quant

custom_ops = {QConv2d: count_convNd}


output_list = []
def forward_hook(module, input, output):
    output_list.append(output[0].cpu().data)


def main():
    config.save = 'ckpt/bn_output'
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if os.path.exists(os.path.join(config.load_path, 'arch.pt')):
        state = torch.load(os.path.join(config.load_path, 'arch.pt'))
        alpha = state['alpha']
    else:
        # print('No arch.pt')
        # sys.exit()
        alpha = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES))
        alpha[:,0] = 10

    # Model #######################################
    model = FBNet_Infer(alpha, config=config)

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    model = torch.nn.DataParallel(model).cuda()

    # if type(config.pretrain) == str:
    #     state_dict = torch.load(config.pretrain)

    #     for key in state_dict.copy():
    #         if 'bn.0' in key:
    #             new_key_list = []

    #             for i in range(1, len(config.num_bits_list)):
    #                 new_key = []
    #                 new_key.extend(key.split('.')[:-2])
    #                 new_key.append(str(i))
    #                 new_key.append(key.split('.')[-1])
    #                 new_key = '.'.join(new_key)

    #                 state_dict[new_key] = state_dict[key]

    #     model.load_state_dict(state_dict, strict=False)

    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = 'ckpt/finetune/weights_199.pt'


    # data loader ############################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if config.dataset == 'cifar10':
        train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
    else:
        print('Wrong dataset.')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        pin_memory=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=4)


    iterator = iter(test_loader)
    input, target = iterator.next()
    input_var = Variable(input, volatile=True).cuda()
    target_var = Variable(target, volatile=True).cuda()

    logging.info('Before BN Calibration')

    activation_list_all = []

    global output_list

    for num_bits in config.num_bits_list:
        hook_list = []
        for module in list(model.modules()):
            if isinstance(module, nn.Conv2d):
                hook_list.append(module.register_forward_hook(forward_hook))

        output_list = []

        model.eval()
        with torch.no_grad():
            model(input_var, num_bits)
        activation_list_all.append(output_list.copy())

        for hook in hook_list:
            hook.remove()

    activation_list_fp = activation_list_all[-1]

    diff_list_all_before = []
    for activation_list in activation_list_all:
        assert len(activation_list) == len(activation_list_fp)

        diff_list = []
        for layer_id, activation in enumerate(activation_list):
            diff_list.append(torch.nn.MSELoss()(activation, activation_list_fp[layer_id]).item())
        diff_list_all_before.append(diff_list)


    logging.info('After BN Calibration')
    activation_list_all = []

    for num_bits in config.num_bits_list:
        for hook in hook_list:
            hook.remove()

        bn_update(train_loader, model, num_bits=num_bits)

        hook_list = []
        for module in list(model.modules()):
            if isinstance(module, nn.Conv2d):
                hook_list.append(module.register_forward_hook(forward_hook))

        output_list = []

        model.eval()
        with torch.no_grad():
            model(input_var, num_bits)
        activation_list_all.append(output_list.copy())


    activation_list_fp = activation_list_all[-1]

    diff_list_all_after = []
    for activation_list in activation_list_all:
        assert len(activation_list) == len(activation_list_fp)

        diff_list = []
        for layer_id, activation in enumerate(activation_list):
            diff_list.append(torch.nn.MSELoss()(activation, activation_list_fp[layer_id]).item())
        diff_list_all_after.append(diff_list)


    diff_final = {'before': diff_list_all_before, 'after':diff_list_all_after}
    np.save('bn_output.npy', diff_final)



if __name__ == '__main__':
    main() 
