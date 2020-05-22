from random import random, randint,shuffle
import numpy as np
import test_for_eyeriss as simnas
import time
from itertools import combinations,permutations
import copy
from  multiprocessing import Queue
import multiprocessing
import math
from ev_util import *
from ev_dict_object import *
#for saving np to matlab 
import scipy.io as sio

import os
import sys
import argparse
import json
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits), temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard, y


def dram_variant_looporder(input_lp_order_dram,input_lp_order_sram):
    return None


# def dram_invariant_looporder(input_lp_order):
#     # input_lp_order:[range(0,4),                                                                           ]
#     #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
#     if not len(input_lp_order[1:])==len(set(input_lp_order[1:])):
#         raise Exception('Please provide lp_order with no duplicate elements')
#     input_rf=rf_noc_template[input_lp_order[0]]
#     lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
#     lp_order_template=['ref_gb_we','ch_out_gb', 'ref_gb_in','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb','ref_gb_out']
#     lp_order_string=[]
#     input_actions=input_lp_order[1:12]
#     for i in range(len(lp_order_template)):
#         lp_order_string.append(lp_order_template[input_actions[i]])
#     index_lst=list(range(len(lp_order_template_dram)))
#     shuffle(index_lst)
#     for i in index_lst:
#         lp_order_string.append(lp_order_template_dram[i])
#     return copy.deepcopy(input_rf)+copy.deepcopy(lp_order_string)


# def tiling_translation(tiling_scheme, tiling_pool, alloc_slots, pe_array, space_partition):
#     tiling_pool=copy.deepcopy(tiling_pool[alloc_slots[pe_array]:alloc_slots[pe_array+1]])
#     index=0
#     for i in range(len(tiling_scheme)):
#         tmp=tiling_scheme[i]
#         for j in range(i+1,len(tiling_scheme)):
#             tmp*=space_partition[pe_array][j]
#         index+=tmp
#     #print('abs index: ',index)
#     tiling_string=tiling_pool[index]
#     return tiling_string[0]


def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=rf_noc_template[pe_array]
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']
    lp_order_string=[]
    for i in range(len(lp_order_template)):
        lp_order_string.append(lp_order_template[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(input_rf)+copy.deepcopy(lp_order_string)


def tiling_translation(tiling_scheme,tiling_pool,alloc_slots,pe_array,space_partition):
    tiling_pool=copy.deepcopy(tiling_pool[alloc_slots[pe_array]:alloc_slots[pe_array+1]])
    index=0
    for i in range(len(tiling_scheme)):
        tmp=tiling_scheme[i]
        for j in range(i+1,len(tiling_scheme)):
            tmp*=space_partition[pe_array][j]
        index+=tmp
    #print('abs index: ',index)
    tiling_string=tiling_pool[index]
    return tiling_string[0]


def dsp_check(tiling_scheme_string,dsp_limit):
    dsp_consumption=1    
    for i in tiling_scheme_string:
        if 'noc' in i:
            dsp_consumption*=tiling_scheme_string[i]
    return dsp_consumption<dsp_limit

def dsp_penalty(tiling_scheme_string,dsp_limit):
    dsp_consumption=1    
    for i in tiling_scheme_string:
        if 'noc' in i:
            dsp_consumption *= tiling_scheme_string[i]

    if dsp_consumption < dsp_limit:
        return 0
    else:
        return dsp_consumption - dsp_limit


def get_fixed_pe_array(num_pe_array=4, np_random_seed=10):
    np.random.seed(np_random_seed)
    pe_array = np.random.randint(num_pe_array)
    np.random.seed(int(time.time()))
    # print('Fix pe array:', pe_array)
    return pe_array


def get_fixed_loop_order(num_loop_order=7, np_random_seed=10):
    np.random.seed(np_random_seed)
    tmp_list = np.arange(num_loop_order)
    np.random.shuffle(tmp_list)
    tmp_list = list(tmp_list)
    np.random.seed(int(time.time()))
    # print('Fix loop order:', tmp_list)
    return tmp_list


def get_fixed_pe_array_dim(num_pe_array_dim=10, np_random_seed=10):
    np.random.seed(np_random_seed)
    pe_array_dim_choices = np.random.randint(num_pe_array_dim)
    np.random.seed(int(time.time()))
    # print('Fix pe array:', pe_array_dim_choices)
    return pe_array_dim_choices


def get_fixed_tiling_factor(partitioned_choices, np_random_seed=10):
    np.random.seed(np_random_seed)

    tiling_choices=[] 
    for i in partitioned_choices:
        tiling_choices.append(np.random.randint(i))

    np.random.seed(int(time.time()))
    # print('Fix tiling factors:', tiling_choices)
    return tiling_choices


def get_fixed_tiling_order(partitioned_choices, np_random_seed=10):
    np.random.seed(np_random_seed)

    tiling_choices_order=[] 
    for _ in partitioned_choices:
        tiling_choices_order.append(np.random.randint(1))

    np.random.seed(int(time.time()))
    # print('Fix tiling factors:', tiling_choices_order)
    return tiling_choices_order


def adjust_temp(args, _epoch, decay=0.9, freq=1000):
    return args.temp * decay ** (_epoch//freq)


def set_mask(param, mask, neg_value=-1e10):
    assert len(param) == len(mask), str(len(param))+str(' ')+str(len(mask))

    param_clone = param.clone()
    for i, flag in enumerate(mask):
        if not flag:
            param_clone.index_put_((torch.tensor(i),), torch.tensor(float(neg_value)))

    return param_clone


def check_num_options_mask(alpha_mask, beta_dram_list_mask, beta_gb_list_mask, omega_mask, gamma_list_mask, lambda_list_mask):
    print('\n==================')
    print('Num of Options:')
    print('pe array:', len([i for i in alpha_mask if i]))
    print('dram loop order:', [len([i for i in beta_mask if i]) for beta_mask in beta_dram_list_mask])
    print('global buffer loop order:', [len([i for i in beta_mask if i]) for beta_mask in beta_gb_list_mask])
    print('pe array dim:', len([i for i in omega_mask if i]))
    print('tiling factor:', [[[len([i for i in gamma_mask if i]) for gamma_mask in sub_sub_list]for sub_sub_list in sub_list] for sub_list in gamma_list_mask])
    print('tiling order:', [[[len([i for i in lambda_mask if i]) for lambda_mask in sub_sub_list]for sub_sub_list in sub_list] for sub_list in lambda_list_mask])
    print('\n==================')


tmp_hw_spec={\
    'gb_vol':2*1024*1024, \
    'rf_vol':512, \
    'num_pe':144, \
    'num_rf':144
}


num_pe_array = 4
num_loop_order = 7
num_pe_array_dim = 10
num_order = 2
layer = 0


def build_arch_parameters(tiling1):
    params = []

    alpha = nn.Parameter(Variable(1e-3*torch.ones(num_pe_array).cuda(), requires_grad=True))
    params.append(alpha)

    beta_dram_list = []
    for i in range(num_loop_order):
        beta = nn.Parameter(Variable(1e-3*torch.ones(num_loop_order).cuda(), requires_grad=True))
        beta_dram_list.append(beta)
        params.append(beta)

    beta_gb_list = []
    for i in range(num_loop_order):
        beta = nn.Parameter(Variable(1e-3*torch.ones(num_loop_order).cuda(), requires_grad=True))
        beta_gb_list.append(beta)
        params.append(beta)

    omega = nn.Parameter(Variable(1e-3*torch.ones(num_pe_array_dim).cuda(), requires_grad=True))
    params.append(omega)

    gamma_list = []
    for pe_array in range(num_pe_array):
        sub_list = []
        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim)
            
            for num_choice in tiling_space_1:
                gamma = nn.Parameter(Variable(1e-3*torch.ones(num_choice).cuda(), requires_grad=True))
                sub_sub_list.append(gamma)
                params.append(gamma)

            sub_list.append(sub_sub_list)

        gamma_list.append(sub_list)

    lambda_list = []
    for pe_array in range(num_pe_array):
        sub_list = []

        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim) 
            
            for num_choice in tiling_space_1:
                order = nn.Parameter(Variable(1e-3*torch.ones(num_order).cuda(), requires_grad=True))
                sub_sub_list.append(order)
                params.append(order)

            sub_list.append(sub_sub_list)

        lambda_list.append(sub_list)

    return params, alpha, beta_dram_list, beta_gb_list, gamma_list, omega, lambda_list


def build_counter(tiling1):
    params = []

    alpha_cnt = np.zeros(num_pe_array)

    beta_dram_list_cnt = []
    for i in range(num_loop_order):
        beta_cnt = np.zeros(num_loop_order)
        beta_dram_list_cnt.append(beta_cnt)

    beta_gb_list_cnt = []
    for i in range(num_loop_order):
        beta_cnt = np.zeros(num_loop_order)
        beta_gb_list_cnt.append(beta_cnt)

    omega_cnt = np.zeros(num_pe_array_dim)

    gamma_list_cnt = []
    for pe_array in range(num_pe_array):
        sub_list = []
        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim) 
            
            for num_choice in tiling_space_1:
                gamma_cnt = np.zeros(num_choice)
                sub_sub_list.append(gamma_cnt)

            sub_list.append(sub_sub_list)

        gamma_list_cnt.append(sub_list)

    lambda_list_cnt = []
    for pe_array in range(num_pe_array):
        sub_list = []

        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim) 
            
            for num_choice in tiling_space_1:
                order_cnt = np.zeros(num_order)
                sub_sub_list.append(order_cnt)

            sub_list.append(sub_sub_list)

        lambda_list_cnt.append(sub_list)

    return alpha_cnt, beta_dram_list_cnt, beta_gb_list_cnt, gamma_list_cnt, omega_cnt, lambda_list_cnt


# input_dnn=[\
#[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[8,0],'row_out':[8,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\

# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[8,0],'row_out':[8,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
# ]


#fpga dedicated 706
# tmp_hw_spec={\
#     'gb_vol':16*1024*1024, \
#     'rf_vol':512*8, \
#     'num_pe':824, \
#     'num_rf':824
# }


def get_hw_efficiency(input_dnn, name, group=1):
    # parser = argparse.ArgumentParser(description="Differentiable Hardware Search")
    # parser.add_argument('--epoch', type=int, default=500,
    #                     help='num of epochs')
    # parser.add_argument('--trial', type=int, default=3,
    #                     help='num of trials')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum of optimizer')
    # parser.add_argument('--lr', type=float, default=1e-11,
    #                     help='initial learning rate')
    # parser.add_argument('--temp', type=float, default=1,
    #                     help='temperature in gumble softmax')
    # parser.add_argument('--adjust_temp', action='store_true', default=False,
    #                     help='whether to adjust the temperature')
    # parser.add_argument('--temp_decay', type=float, default=0.9,
    #                     help='decay of temperature')
    # parser.add_argument('--temp_freq', type=float, default=1000,
    #                     help='frequency of per temperature decay')
    # parser.add_argument('--random_seed', type=int, default=10,
    #                     help='random seed for np.random')
    # parser.add_argument('--kl_loss', type=float, default=0,
    #                     help='whether to push the choices to uniform distribution with KL loss')
    # parser.add_argument('--entropy_loss', type=float, default=0,
    #                     help='whether to push the choices to uniform distribution with entropy loss')
    # parser.add_argument('--limit', type=float, default=None,
    #                     help='metric limit')
    # parser.add_argument('--pruning_thres', type=float, default=None,
    #                     help='pruning threshold')
    # parser.add_argument('--pruning_ratio', type=float, default=0.5,
    #                     help='pruning ratio')
    # parser.add_argument('--pruning_epoch', type=int, default=None,
    #                     help='pruning epoch')
    # parser.add_argument('--fix_pe_array', action='store_true', default=False,
    #                     help='whether to fix the pe array')
    # parser.add_argument('--fix_pe_array_dim', action='store_true', default=False,
    #                     help='whether to fix the pe array dim choice')
    # parser.add_argument('--fix_loop_order_dram', action='store_true', default=False,
    #                     help='whether to fix the loop order of dram')
    # parser.add_argument('--fix_loop_order_gb', action='store_true', default=False,
    #                     help='whether to fix the loop order of global buffer')
    # parser.add_argument('--fix_tiling_factor', action='store_true', default=False,
    #                     help='whether fix the tiling factors')
    # parser.add_argument('--fix_tiling_order', action='store_true', default=False,
    #                     help='whether fix the tiling orders')
    # parser.add_argument('--final_infer', action='store_true', default=False,
    #                     help='whether conduct final inference')
    # parser.add_argument('--save_path', type=str, default='trial_info.json',
    #                     help='save path of trial info')
    # args = parser.parse_args()


    ############################
    #user interface
    ############################

    #generate the design space of all possible tiling factors
    #the space is partitioned according to alloc_slots based on the rf_noc_template choice (PE array)
    print('Building Tiling Pool...')
    # (tiling_pool,alloc_slots,space_partition) = fpga_tiling_generator(input_dnn,tmp_hw_spec['gb_vol'],tmp_hw_spec['num_pe'],return_partitioned_space=True)
    # print(tiling_pool[0])
    # print(len(tiling_pool))
    # print(alloc_slots)
    # print(space_partition)

    stride = input_dnn[0][0]

    global num_pe_array_dim

    if group > 2:
        num_pe_array_dim = 1
    else:
        num_pe_array_dim = 10

    if group > 1:
        input_dnn[0][1]['ch_out'] = [int(input_dnn[0][1]['ch_out'][0]/group), 0]
        input_dnn[0][1]['ch_in'] = [int(input_dnn[0][1]['ch_in'][0]/group), 0]

    tiling1 = fpga_tiling_generator(input_dnn, tmp_hw_spec)


    criterion = nn.KLDivLoss().cuda()

    # if args.fix_pe_array:
    #     pe_array_fix = get_fixed_pe_array(num_pe_array, args.random_seed)
    # if args.fix_loop_order_dram:
    #     input_lp_order_dram_fix = get_fixed_loop_order(num_loop_order, args.random_seed)
    # if args.fix_loop_order_gb:
    #     input_lp_order_gb_fix = get_fixed_loop_order(num_loop_order, args.random_seed)
    # if args.fix_pe_array_dim:
    #     pe_array_dim_choices_fix = get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)

    # pe_array_fix = get_fixed_pe_array(num_pe_array, args.random_seed)
    # pe_array_dim_choices_fix = get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)
    # tiling_space_1 = tiling1.tiling_space_partition(pe_array_fix, layer, pe_array_dim_choices_fix)
    # if args.fix_tiling_factor:
    #     tiling_choices_fix = get_fixed_tiling_factor(tiling_space_1, args.random_seed)
    # if args.fix_tiling_order:
    #     tiling_choices_order_fix = get_fixed_tiling_order(tiling_space_1, args.random_seed)

    print("Start Trials")

    stop_epoch_list = []
    metric_list = []
    best_metric_list = []
    best_epoch_list = []
    best_hw_list = []

    for _trial in range(args.trial):
        best_metric_single = 1e8
        best_metric = 1e8
        best_epoch = 0
        num_exceed = 0
        tbar = tqdm(range(args.epoch), ncols=100)

        # print('\nBuilding Arch Parameters for Trial', _trial+1)
        params, alpha_orig, beta_dram_list_orig, beta_gb_list_orig, gamma_list_orig, omega_orig, lambda_list_orig = build_arch_parameters(tiling1)

        alpha_cnt, beta_dram_list_cnt, beta_gb_list_cnt, gamma_list_cnt, omega_cnt, lambda_list_cnt = build_counter(tiling1)

        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
        # lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 4000, 6000, 8000], gamma=0.5)
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.1)

        print('\nStart Searching for Trial', _trial+1)

        for _epoch in tbar:
            if args.adjust_temp:
                temp = adjust_temp(args, _epoch, decay=args.temp_decay, freq=args.temp_freq)
            else:
                temp = args.temp

            # pruning search space
            if args.pruning_thres and args.pruning_epoch:
                if _epoch == args.pruning_epoch:
                    alpha_mask = alpha_cnt.argsort() < np.ceil(alpha_cnt.size*args.pruning_ratio)

                    beta_dram_list_mask = []
                    for i in range(len(beta_dram_list_cnt)):
                        mask = beta_dram_list_cnt[i].argsort() < np.ceil(beta_dram_list_cnt[i].size*args.pruning_ratio)
                        beta_dram_list_mask.append(mask)

                    beta_gb_list_mask = []
                    for i in range(len(beta_gb_list_cnt)):
                        mask = beta_gb_list_cnt[i].argsort() < np.ceil(beta_gb_list_cnt[i].size*args.pruning_ratio)
                        beta_gb_list_mask.append(mask)

                    omega_mask = omega_cnt.argsort() < np.ceil(omega_cnt.size*args.pruning_ratio)

                    gamma_list_mask = []
                    for pe_array in range(num_pe_array):
                        sub_list = []
                        for pe_array_dim in range(num_pe_array_dim):
                            sub_sub_list = []

                            for i in range(len(gamma_list_cnt[pe_array][pe_array_dim])):
                                mask = gamma_list_cnt[pe_array][pe_array_dim][i].argsort() < np.ceil(gamma_list_cnt[pe_array][pe_array_dim][i].size*args.pruning_ratio)
                                sub_sub_list.append(mask)

                                # print(gamma_list_cnt[pe_array][pe_array_dim][num_choice])
                                # print(mask)
                                # input()

                            sub_list.append(sub_sub_list)

                        gamma_list_mask.append(sub_list)

                    lambda_list_mask = []
                    for pe_array in range(num_pe_array):
                        sub_list = []
                        for pe_array_dim in range(num_pe_array_dim):
                            sub_sub_list = []
                            for i in range(len(lambda_list_cnt[pe_array][pe_array_dim])):
                                mask = lambda_list_cnt[pe_array][pe_array_dim][i].argsort() < np.ceil(lambda_list_cnt[pe_array][pe_array_dim][i].size*args.pruning_ratio)
                                sub_sub_list.append(mask)

                            sub_list.append(sub_sub_list)

                        lambda_list_mask.append(sub_list)

                    check_num_options_mask(alpha_mask, beta_dram_list_mask, beta_gb_list_mask, omega_mask, gamma_list_mask, lambda_list_mask)


                if _epoch >= args.pruning_epoch:
                    # alpha = set_mask(alpha_orig, mask)
                    alpha = alpha_orig

                    beta_dram_list = []
                    for i in range(len(beta_dram_list_orig)):
                        beta_dram_list.append(set_mask(beta_dram_list_orig[i], beta_dram_list_mask[i]))                    

                    beta_gb_list = []
                    for i in range(len(beta_gb_list_orig)):
                        beta_gb_list.append(set_mask(beta_gb_list_orig[i], beta_gb_list_mask[i])) 

                    omega = omega_orig

                    # print(np.array(gamma_list_orig))
                    # print(np.array(gamma_list_mask))
                    # input()

                    gamma_list = []
                    for i in range(len(gamma_list_orig)):
                        sub_list = []
                        for j in range(len(gamma_list_orig[i])):
                            sub_sub_list = []
                            for k in range(len(gamma_list_orig[i][j])):
                                sub_sub_list.append(set_mask(gamma_list_orig[i][j][k], gamma_list_mask[i][j][k]))
                            sub_list.append(sub_sub_list)
                        gamma_list.append(sub_list)

                    lambda_list = lambda_list_orig
                    # lambda_list = []
                    # for i in range(len(lambda_list_orig)):
                    #     sub_list = []
                    #     for j in range(len(lambda_list_orig[i])):
                    #         sub_sub_list = []
                    #         for k in range(len(lambda_list_orig[i][j])):
                    #             sub_sub_list.append(set_mask(lambda_list_orig[i][j][k], lambda_list_mask[i][j][k]))
                    #         sub_list.append(sub_sub_list)
                    #     lambda_list.append(sub_list)

                else:
                    alpha = alpha_orig
                    beta_dram_list = beta_dram_list_orig
                    beta_gb_list =beta_gb_list_orig
                    gamma_list = gamma_list_orig
                    omega = omega_orig
                    lambda_list = lambda_list_orig

            else:
                alpha = alpha_orig
                beta_dram_list = beta_dram_list_orig
                beta_gb_list =beta_gb_list_orig
                gamma_list = gamma_list_orig
                omega = omega_orig
                lambda_list = lambda_list_orig

            params_diff = []
            params_distrib = []

            if args.fix_pe_array:
                pe_array = pe_array_fix
            else:
                alpha_sample, alpha_distrib = gumbel_softmax(alpha, temperature=temp)
                pe_array = alpha_sample.argmax()
                params_diff.append(alpha_sample[pe_array])
                params_distrib.append(alpha_distrib)

            if args.fix_loop_order_dram:
                input_lp_order_dram = input_lp_order_dram_fix
            else:
                input_lp_order_dram = []
                for i in range(num_loop_order):
                    beta_sample, beta_distrib = gumbel_softmax(beta_dram_list[i], temperature=temp)
                    order_list = beta_sample.argsort(descending=True)

                    for order in order_list:
                        if order not in input_lp_order_dram:
                            break

                    input_lp_order_dram.append(order)
                    params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                    params_distrib.append(beta_distrib)

            if args.fix_loop_order_gb:
                input_lp_order_gb = input_lp_order_gb_fix
            else:
                input_lp_order_gb = []
                for i in range(num_loop_order):
                    beta_sample, beta_distrib = gumbel_softmax(beta_gb_list[i], temperature=temp)
                    order_list = beta_sample.argsort(descending=True)

                    for order in order_list:
                        if order not in input_lp_order_gb:
                            break

                    input_lp_order_gb.append(order)
                    params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                    params_distrib.append(beta_distrib)

            lp_order_string = dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb)

            if args.fix_pe_array_dim:
                pe_array_dim_choices = pe_array_dim_choices_fix
            else:
                omega_sample, omega_distrib = gumbel_softmax(omega, temperature=temp)
                pe_array_dim_choices = omega_sample.argmax()
                params_diff.append(omega_sample[pe_array_dim_choices])
                params_distrib.append(omega_distrib)

            tiling_space_1 = tiling1.tiling_space_partition(pe_array, layer, pe_array_dim_choices)

            if args.fix_tiling_factor:
                tiling_choices = tiling_choices_fix
            else:
                tiling_choices = []
                for i in range(len(gamma_list[pe_array][pe_array_dim_choices])):
                    gamma_sample, gamma_distrib = gumbel_softmax(gamma_list[pe_array][pe_array_dim_choices][i], temperature=temp)
                    factor = gamma_sample.argmax()

                    tiling_choices.append(factor)

                    params_diff.append(gamma_sample[factor])
                    params_distrib.append(gamma_distrib)

            if args.fix_tiling_order:
                tiling_choices_order = tiling_choices_order_fix
            else:
                tiling_choices_order = []
                for i in range(len(lambda_list[pe_array][pe_array_dim_choices])):
                    lambda_sample, lambda_distrib = gumbel_softmax(lambda_list[pe_array][pe_array_dim_choices][i], temperature=temp)
                    factor = lambda_sample.argmax()

                    tiling_choices_order.append(factor)

                    params_diff.append(lambda_sample[factor])
                    params_distrib.append(lambda_distrib)

            tiling_string = tiling1.tiling_translation(layer, pe_array, pe_array_dim_choices, tiling_choices, tiling_choices_order)

            metric, valid = life_eval(tiling_string, stride, tmp_hw_spec, df_order=lp_order_string)
            
            if valid:
                # if group > 1:
                #     num_parallel = compute_parallel(lp_order_string, tiling_string, tmp_hw_spec, stride)
                #     metric = metric * group * group / min(num_parallel, group)
                
                loss = metric 

                if group > 1:
                    if best_metric_single > metric:
                        num_parallel = compute_parallel(lp_order_string, tiling_string, tmp_hw_spec, stride)
                        metric_scale = metric * group * group / min(num_parallel, group)
                    
                        if best_metric > metric_scale:
                            best_metric = metric_scale
                            best_epoch = _epoch
                            best_metric_single = metric

                            best_hw = {'pe_array':pe_array, 'loop_order_dram':input_lp_order_dram, 'loop_order_gb':input_lp_order_gb,
                                    'pe_array_dim': pe_array_dim_choices, 'tiling_factor':tiling_choices, 'tiling_order':tiling_choices_order, 'edp': metric}

                else:
                    if best_metric > metric:
                        best_metric = metric
                        best_epoch = _epoch

                        best_hw = {'pe_array':pe_array, 'loop_order_dram':input_lp_order_dram, 'loop_order_gb':input_lp_order_gb,
                                    'pe_array_dim': pe_array_dim_choices, 'tiling_factor':tiling_choices, 'tiling_order':tiling_choices_order, 'edp': metric}
                                    
                for item in params_diff:
                    # loss = loss * ((1-item).detach() + item)
                    loss = loss * item

                if args.pruning_thres and args.pruning_epoch:
                    if metric > args.pruning_thres:
                        alpha_cnt[pe_array] += 1
                        omega_cnt[pe_array_dim_choices] += 1

                        for i in range(len(beta_dram_list_cnt)):
                            beta_dram_list_cnt[i][input_lp_order_dram[i]] += 1

                        for i in range(len(beta_gb_list_cnt)):
                            beta_gb_list_cnt[i][input_lp_order_gb[i]] += 1

                        for i in range(len(gamma_list_cnt[pe_array][pe_array_dim_choices])):
                            gamma_list_cnt[pe_array][pe_array_dim_choices][i][tiling_choices[i]] += 1

                        for i in range(len(lambda_list_cnt[pe_array][pe_array_dim_choices])):
                            lambda_list_cnt[pe_array][pe_array_dim_choices][i][tiling_choices_order[i]] += 1

                optimizer.zero_grad()
                lr_policy.step()

                if args.kl_loss:
                    for i in range(len(params_distrib)):
                        loss += args.kl_loss * criterion((torch.ones_like(params_distrib[i]).cuda()/params_distrib[i].shape[0]).log(), params_distrib[i])

                if args.entropy_loss:
                    for i in range(len(params_distrib)):
                        loss += args.entropy_loss * torch.mean(params_distrib[i] * params_distrib[i].log())

                loss.backward()
                optimizer.step()

                loss = loss.item()

            else:
                loss = -1
                num_exceed += 1

            tbar.set_description("[Trial %d/%d][Epoch %d/%d] Loss: %.3f Best Metric: %.3f Num Exceed: [%d/%d]" % (_trial+1, args.trial, _epoch + 1, args.epoch, loss, best_metric, num_exceed, _epoch+1))

        best_metric_list.append(best_metric)
        best_epoch_list.append(best_epoch)
        best_hw_list.append(best_epoch)

    if not os.path.exists('searched_hw'):
        os.makedirs('searched_hw')

    with open(os.path.join('searched_hw', name+'.json'), 'w') as f:
        json.dump(best_hw_list[np.array(best_metric_list).argmin()], f)

    print('Best Metric', min(best_metric_list))

    return min(best_metric_list)


if __name__ == '__main__':
    input_dnn = [[1, {'ch_out':[32,0],'ch_in':[32,0],'batch':[1,0],'col_out':[32,0],
                     'row_out':[32,0],'row_kernel':[1, 0],'col_kernel':[1,0]}]]

    get_hw_efficiency(input_dnn, name='hhh', group=32)