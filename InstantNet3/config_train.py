# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('autoai2c')


C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'QANAS'

"""Data Dir and Weight Dir"""
C.dataset_path = "/data1/ILSVRC/Data/CLS-LOC" # Specify path to ImageNet-1000

C.world_size = 1  # num of nodes
C.multiprocessing_distributed = True
C.rank = 0  # node rank
C.dist_backend = 'nccl'
C.dist_url = 'tcp://eic-2019gpu3.ece.rice.edu:10001'  # url used to set up distributed training
# C.dist_url = 'tcp://127.0.0.1:10001'

C.num_workers = 4  # workers per gpu
C.batch_size = 256

C.gpu = None

C.dataset = 'imagenet'

if C.dataset == 'cifar10':
    C.num_classes = 10
elif C.dataset == 'cifar100':
    C.num_classes = 100
elif C.dataset == 'imagenet':
    C.num_classes = 1000
else:
    print('Wrong dataset.')
    sys.exit()

"""Image Config"""

C.num_train_imgs = 1281167
C.num_eval_imgs = 50000

""" Settings for network, this would be different for each kind of model"""
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Train Config"""


C.opt = 'Sgd'

C.momentum = 0.9
C.weight_decay = 5e-4

C.betas=(0.5, 0.999)


""" Search Config """
C.grad_clip = 5

C.pretrain = False
# C.pretrain = 'ckpt/finetune/weights_199.pt'

# C.bit_schedule = 'high2low'
# C.bit_schedule = 'low2high'
# C.bit_schedule = 'sandwich'
C.bit_schedule = 'avg_loss'
# C.bit_schedule = 'max_loss'

C.dws_chwise_quant = True

### supernet definition ###
C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
C.stride_list = [1, 2, 2, 2, 1, 2, 1]

C.stem_channel = 16
C.header_channel = 1984


C.num_bits_list = [4, 5, 6, 8]

# C.loss_scale = [8, 4, 8/3, 2, 1]
C.loss_scale = [1, 1, 1, 1, 1]

C.distill_weight = 1

C.cascad = True

C.update_bn_freq = None

C.num_bits_list_schedule = None
# C.choice_list_schedule = 'high2low'
# C.choice_list_schedule = 'low2high'

C.schedule_freq = 20

########################################

C.niters_per_epoch = C.num_train_imgs // C.batch_size
C.image_height = 224 # this size is after down_sampling
C.image_width = 224

C.save = "finetune"
########################################

C.nepochs = 360

C.eval_epoch = 1

# C.lr_schedule = 'multistep'
# C.lr = 1e-1

C.lr_schedule = 'multistep'
C.lr = 0.1

# linear 
C.decay_epoch = 100
# exponential
C.lr_decay = 0.97
# multistep
C.milestones = [90, 180, 270]
C.gamma = 0.1
# cosine
C.learning_rate_min = 0

C.load_path = 'ckpt/search'

C.eval_only = False
C.update_bn = True
C.show_distrib = True

C.finetune_bn = False
C.ft_bn_epoch = 10
C.ft_bn_lr = 1e-3
C.ft_bn_momentum = 0.9

