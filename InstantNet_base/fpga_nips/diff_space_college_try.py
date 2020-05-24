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
#for saving np to matlab 
import scipy.io as sio
from sympy.solvers import solve
from sympy import Symbol

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



def dram_invariant_looporder(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................
    
    input_lp_order_dram=copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb=copy.deepcopy(input_lp_order_gb)
    if not (len(input_lp_order_gb)==len(set(input_lp_order_gb)) and len(input_lp_order_dram)==len(set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf=copy.deepcopy(noc_template[pe_array])
    lp_order_template_dram=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
    lp_order_template_gb=['ch_out_gb','ch_in_gb','col_kernel_gb', 'row_out_gb','batch_gb','col_out_gb','row_kernel_gb']

    lp_order_string=[]
    lp_order_string+=input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


def dram_invariant_looporder_dw(pe_array, input_lp_order_dram, input_lp_order_gb):
    # input_lp_order:[range(0,4),                                                                           ]
    #                 pe_array  ,1st pos   ,2nd pos   , 3rd pos  , .........................................

    input_lp_order_dram = copy.deepcopy(input_lp_order_dram)
    input_lp_order_gb = copy.deepcopy(input_lp_order_gb)

    if not (len(input_lp_order_gb) == len(set(input_lp_order_gb)) and len(input_lp_order_dram) == len(
            set(input_lp_order_dram))):
        raise Exception('Please provide lp_order with no duplicate elements')
    input_rf = copy.deepcopy(noc_template_dw[pe_array])
    lp_order_template_dram = ['col_out_dram', 'ch_out_dram', 'batch_dram', 'row_out_dram',
                              'col_kernel_dram', 'row_kernel_dram']
    lp_order_template_gb = ['ch_out_gb', 'col_kernel_gb', 'row_out_gb', 'batch_gb', 'col_out_gb',
                            'row_kernel_gb']


    lp_order_string = []
    lp_order_string += input_rf
    for i in range(len(lp_order_template_gb)):
        lp_order_string.append(lp_order_template_gb[input_lp_order_gb[i]])
    for i in range(len(lp_order_template_dram)):
        lp_order_string.append(lp_order_template_dram[input_lp_order_dram[i]])
    return copy.deepcopy(lp_order_string)


# input_dnn=[\
# [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
# ]



# tiling1=asic_tiling_generator(input_dnn,hw_spec)
# print(tiling1.rs2_rf_gb_tiling_choices_num[5][5])
# print(tiling1.tiling_translation(5,1,5,[7,9,0,4,0,0,0],[0,1,2,3,4,1,1]))
# exit()

############################
#user interface
############################


# def tiling_generator(input_dnn,tmp_hw_spec,bw=16):
#     choices = {'ch_in': [], 'ch_out': [], 'col_kernel': [], 'row_kernel': [], 'col_out': [], 'row_out': [], 'batch': []}
#     for layer in input_dnn:
#         choices['ch_in']+=r_factors(layer[1]['ch_in'][0])
#         choices['ch_out'] += r_factors(layer[1]['ch_out'][0])
#         choices['col_kernel'] += r_factors(layer[1]['col_kernel'][0])
#         choices['row_kernel'] += r_factors(layer[1]['row_kernel'][0])
#         choices['col_out'] += r_factors(layer[1]['col_out'][0])
#         choices['row_out'] += r_factors(layer[1]['row_out'][0])
#         choices['batch'] += r_factors(layer[1]['batch'][0])
#     for i in choices:
#         choices[i]=set(choices[i])
#     choices_len_rf=[]
#     choices_len_gb=[]
#     largest_rf=tmp_hw_spec["rf_vol"]/bw
#     largest_gb=tmp_hw_spec["gb_vol"]/bw/100
#     for i in choices:
#         rf_bound=0
#         gb_bound=0
#         for syze in choices[i]:
#             if largest_rf> syze:
#                 rf_bound+=1
#             if largest_gb> syze:
#                 gb_bound+=1
#         choices_len_rf.append(rf_bound)
#         choices_len_gb.append(gb_bound)
#
#
#     return choices, choices_len_rf, choices_len_gb
# [choices, choices_len_rf, choices_len_gb]=tiling_generator(input_dnn,tmp_hw_spec)
#
#
#
#
#
#
# def pe_array_dimention_optimizer(input_dnn,tmp_hw_spec, range_c=0.9, consideration_range=10):
#     #the number of tiles needed as penalty
#     num_pe=tmp_hw_spec['num_pe']
#     dim_3=[]
#     dim_4=[]
#     for i in range(int(range_c*num_pe),num_pe):
#         dim_3+=factor_n(i,3)
#         dim_4+=factor_n(i,4)
#     dim_4=permute_factor(dim_4)
#     dim_3=permute_factor(dim_3)
#     pe_array_pool={}
#     pe_array_pool[0]=[]
#     pe_array_pool[1]=[]
#     pe_array_pool[2]=[]
#     pe_array_pool[3] = []
#
#     score_board=[]
#     #pe array 1
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['row_kernel'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) + math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[0]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#
#
#     score_board=[]
#     #pe array 2
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['col_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[1]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#
#     score_board=[]
#     #pe array 2
#     for i in dim_4:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['row_kernel'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_in'][0]/i[2]) +    math.ceil(layer[1]['ch_out'][0]/i[3]))
#         score_board.append(score)
#     pe_array_pool[2]+=[i[1] for i in sorted(zip(score_board,dim_4))][0:consideration_range]
#
#     score_board=[]
#     #pe array 3
#     for i in dim_3:
#         score=0
#         for layer in input_dnn:
#             score+=(math.ceil(layer[1]['row_out'][0]/i[0])+ math.ceil(layer[1]['col_out'][0]/i[1])\
#                     +math.ceil(layer[1]['ch_out'][0]/i[2]))
#         score_board.append(score)
#     pe_array_pool[3]+=[i[1] for i in sorted(zip(score_board,dim_3))][0:consideration_range]
#
#
#
#     return pe_array_pool


def hardware_translation(ratio_noc,ratio_gb,pe_array,tmp_hw_spec,bw=16):
    # print('\n std:', ratio_noc,ratio_gb,pe_array)

    #in_ch, out_ch, X, Y, X_K, Y_K
    rf_vol=tmp_hw_spec['rf_vol']/bw
    gb_vol=tmp_hw_spec['gb_vol']/bw
    pe_num=tmp_hw_spec['num_pe']
    #calculate rf
    consumption_dict={}
    consumption_dict["ch_in_rf"]=1
    consumption_dict["ch_out_rf"] = 1
    consumption_dict["col_out_rf"] = 1
    consumption_dict["row_out_rf"] =1
    consumption_dict["col_kernel_rf"] =1
    consumption_dict["row_kernel_rf"] = 1
    scaling_ratio=1
    #calculate pe
    if pe_array==3:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_out_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio=(pe_num/(consumption_dict['row_out_noc']*consumption_dict['col_out_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array==0:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2]/ratio_noc[3])**(1/4)
        consumption_dict['col_kernel_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y = (pe_num / ratio_noc[0] / ratio_noc[1] / ratio_noc[2] / ratio_noc[3]) ** (1 / 4)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_in_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[3] ), 1)
        scaling_ratio = (pe_num /( consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] * consumption_dict['ch_in_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    #calculate gb

    in_rf_consumption=consumption_dict["ch_in_rf"]*(consumption_dict["col_out_rf"]+consumption_dict['col_kernel_rf']-1)*(consumption_dict["row_out_rf"]+consumption_dict['row_kernel_rf']-1)
    out_rf_consumption=consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption=consumption_dict["ch_in_rf"] * consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"]*consumption_dict["row_kernel_rf"]
    #print((in_rf_consumption +out_rf_consumption+we_rf_consumption)*16)
    in_rf_consumption_for_all_pes=in_rf_consumption
    out_rf_consumption_for_all_pes=out_rf_consumption
    we_rf_consumption_for_all_pes=we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_in' in i:
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif 'ch_out' in i:
                out_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass
    a=(ratio_gb[0]*ratio_gb[2]*ratio_gb[3])*in_rf_consumption_for_all_pes+(ratio_gb[1] * ratio_gb[2] * ratio_gb[3])*out_rf_consumption_for_all_pes
    b=(ratio_gb[0] * ratio_gb[1] * ratio_gb[4]*ratio_gb[5])*we_rf_consumption_for_all_pes
    roots=np.roots([b,a,0,0,-gb_vol])
    roots=roots[np.isreal(roots)]
    z=0
    for i in roots:
        if float(i.real)>0:
            z=i.real
    consumption_dict["ch_in_gb"]=math.floor(max(ratio_gb[0]*z,1))
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[1] * z,1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[2] * z,1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[3] * z,1))
    consumption_dict["col_kernel_gb"] =math.floor( max(ratio_gb[4] * z,1))
    consumption_dict["row_kernel_gb"] =math.floor( max(ratio_gb[5] * z,1))
    scaling_ratio=(gb_vol/(consumption_dict["ch_in_gb"]*(consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1)*(consumption_dict["col_out_gb"]+consumption_dict["col_kernel_gb"]-1)*in_rf_consumption_for_all_pes+consumption_dict["ch_out_gb"]*consumption_dict["row_out_gb"]*consumption_dict["col_out_gb"]*out_rf_consumption_for_all_pes+ \
                          consumption_dict["ch_in_gb"]*consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))
    
    # print('consumption_dict_std',consumption_dict)

    while scaling_ratio<1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)]=consumption_dict[i]
        if len(components_need_scaled)==sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -=1
            consumption_dict[i] =max(consumption_dict[i],0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_in_gb"] * (consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1) * (consumption_dict["col_out_gb"]+consumption_dict["col_kernel_gb"]-1) * in_rf_consumption_for_all_pes + consumption_dict["ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict["col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_in_gb"] * consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))



    return consumption_dict



def hardware_translation_dw(ratio_noc,ratio_gb,pe_array,tmp_hw_spec,bw=16):
    # print('\n dw:', ratio_noc,ratio_gb,pe_array)
    #out_ch, X, Y, X_K, Y_K
    rf_vol=tmp_hw_spec['rf_vol']/bw
    gb_vol=tmp_hw_spec['gb_vol']/bw
    pe_num=tmp_hw_spec['num_pe']
    consumption_dict={}
    consumption_dict["ch_out_rf"] = 1
    consumption_dict["col_out_rf"] = 1
    consumption_dict["row_out_rf"] = 1
    consumption_dict["col_kernel_rf"] = 1
    consumption_dict["row_kernel_rf"] = 1
    #calculate pe
    if pe_array==3:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_out_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio=(pe_num/(consumption_dict['row_out_noc']*consumption_dict['col_out_noc']*consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array==0:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['col_kernel_noc']=max(math.floor(y*ratio_noc[0]),1)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[1] * y), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio =(pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['row_kernel_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 1:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['col_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio =(pe_num / (consumption_dict['col_kernel_noc'] * consumption_dict['col_out_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    elif pe_array == 2:
        y=(pe_num/ratio_noc[0]/ratio_noc[1]/ratio_noc[2])**(1/3)
        consumption_dict['row_kernel_noc'] = max(math.floor(y * ratio_noc[0] ), 1)
        consumption_dict['col_out_noc'] = max(math.floor(y * ratio_noc[1] ), 1)
        consumption_dict['ch_out_noc'] = max(math.floor(y * ratio_noc[2] ), 1)
        scaling_ratio = (pe_num / (consumption_dict['row_kernel_noc'] * consumption_dict['col_out_noc'] *consumption_dict['ch_out_noc']))
        if scaling_ratio<1:
            components_need_scaled=[]
            for i in consumption_dict:
                    if "noc" in i:
                        if consumption_dict[i]!=1:
                            components_need_scaled.append(str(i))
            for i in components_need_scaled:
                consumption_dict[i] = math.floor(consumption_dict[i] * (scaling_ratio ** (1 / len(components_need_scaled))))
    #calculate gb
    in_rf_consumption=consumption_dict["ch_out_rf"]*(consumption_dict["col_out_rf"]+consumption_dict['col_kernel_rf']-1)*(consumption_dict["row_out_rf"]+consumption_dict['row_kernel_rf']-1)
    out_rf_consumption=consumption_dict["ch_out_rf"] * consumption_dict["col_out_rf"] * consumption_dict["row_out_rf"]
    we_rf_consumption= consumption_dict["ch_out_rf"] * consumption_dict["col_kernel_rf"]*consumption_dict["row_kernel_rf"]
    #print((in_rf_consumption +out_rf_consumption+we_rf_consumption)*16)
    in_rf_consumption_for_all_pes=in_rf_consumption
    out_rf_consumption_for_all_pes=out_rf_consumption
    we_rf_consumption_for_all_pes=we_rf_consumption
    for i in consumption_dict:
        if 'noc' in i:
            if 'ch_out' in i:
                in_rf_consumption_for_all_pes *= consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
                we_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('col_out' in i) or ('row_out' in i):
                in_rf_consumption_for_all_pes*=consumption_dict[i]
                out_rf_consumption_for_all_pes*=consumption_dict[i]
            elif ('row_kernel' in i) or ('col_kernel' in i):
                we_rf_consumption_for_all_pes *= consumption_dict[i]
            else:
                pass

    a=(ratio_gb[0]*ratio_gb[1]*ratio_gb[2])*in_rf_consumption_for_all_pes+(ratio_gb[0] * ratio_gb[1] * ratio_gb[2])*out_rf_consumption_for_all_pes
    b=(ratio_gb[0] * ratio_gb[3]*ratio_gb[4])*we_rf_consumption_for_all_pes
    
    # roots=np.roots([b,a,0,0,-gb_vol])
    # roots=roots[np.isreal(roots)]
    # z=0
    # for i in roots:
    #     if float(i.real)>0:
    #         z=i.real

    z=(gb_vol/(a+b))**(1/3)
    consumption_dict["ch_out_gb"] = math.floor(max(ratio_gb[0] * z,1))
    consumption_dict["col_out_gb"] = math.floor(max(ratio_gb[1] * z,1))
    consumption_dict["row_out_gb"] = math.floor(max(ratio_gb[2] * z,1))
    consumption_dict["col_kernel_gb"] =math.floor( max(ratio_gb[3] * z,1))
    consumption_dict["row_kernel_gb"] =math.floor( max(ratio_gb[4] * z,1))
    scaling_ratio=(gb_vol/(consumption_dict["ch_out_gb"]*(consumption_dict["row_out_gb"]+consumption_dict['row_kernel_gb']-1)*(consumption_dict["col_out_gb"]+consumption_dict['col_kernel_gb']-1)*in_rf_consumption_for_all_pes+consumption_dict["ch_out_gb"]*consumption_dict["row_out_gb"]*consumption_dict["col_out_gb"]*out_rf_consumption_for_all_pes+ \
                          consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes))**(1/3)
    
    # print('consumption_dict_dw', consumption_dict)

    while scaling_ratio<1:
        components_need_scaled = {}
        for i in consumption_dict:
            if ("gb" in i):
                if consumption_dict[i] != 1:
                    components_need_scaled[str(i)]=consumption_dict[i]
        if len(components_need_scaled)==sum(components_need_scaled.values()):
            print('can not fit the gb requirement')
            raise
        for i in components_need_scaled:
            consumption_dict[i] -=1
            consumption_dict[i] =max(consumption_dict[i],0)
        scaling_ratio = (gb_vol / (consumption_dict["ch_out_gb"] * (consumption_dict["row_out_gb"] + consumption_dict['row_kernel_gb'] - 1) * (consumption_dict["col_out_gb"] + consumption_dict['col_kernel_gb'] - 1) * in_rf_consumption_for_all_pes + consumption_dict["ch_out_gb"] * consumption_dict["row_out_gb"] * consumption_dict["col_out_gb"] * out_rf_consumption_for_all_pes + \
                                   consumption_dict["ch_out_gb"] * consumption_dict["col_kernel_gb"] * consumption_dict["row_kernel_gb"] * we_rf_consumption_for_all_pes)) ** (1 / 3)


    return consumption_dict



def tiling_translation( consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i]=min(consumption_dict[i],layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i]=min(consumption_dict[i],math.ceil(layer[1][str(i)[:-4]][0]/tiling_str[-1][str(i)[:-4]+"_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"]/tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf']=tiling_str[-1]['batch_gb']=1
        consumption_dict['batch_rf']=consumption_dict['batch_gb']=1
        dram_list=['col_out_dram', 'ch_out_dram', 'batch_dram','ch_in_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        for i in dram_list:
            consumption_dict[i]=1
            try:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][str(i)[:-5] + "_noc"] / tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except:
                raise

    # print('tiling_str', tiling_str)
    return tiling_str,consumption_dict


def tiling_translation_dw(consumption_dict, input_dnn):
    tiling_str = []
    for layer in input_dnn:
        tiling_str.append({})
        for i in consumption_dict:
            if "rf" in i:
                tiling_str[-1][i]=min(consumption_dict[i],layer[1][str(i)[:-3]][0])
        for i in consumption_dict:
            if "noc" in i:
                tiling_str[-1][i]=min(consumption_dict[i],math.ceil(layer[1][str(i)[:-4]][0]/tiling_str[-1][str(i)[:-4]+"_rf"]))
        for i in consumption_dict:
            if "gb" in i:
                try:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                        math.ceil(layer[1][str(i)[:-3]][0] / tiling_str[-1][str(i)[:-3] + "_noc"]/tiling_str[-1][str(i)[:-3] + "_rf"]))
                except KeyError:
                    tiling_str[-1][i] = min(consumption_dict[i],
                                            math.ceil(layer[1][str(i)[:-3]][0] /
                                                      tiling_str[-1][str(i)[:-3] + "_rf"]))
                except:
                    raise
        tiling_str[-1]['batch_rf']=tiling_str[-1]['batch_gb']=1
        consumption_dict['batch_rf']=consumption_dict['batch_gb']=1
        dram_list=['col_out_dram', 'ch_out_dram', 'batch_dram','row_out_dram','col_kernel_dram','row_kernel_dram']
        for i in dram_list:
            consumption_dict[i]=1
            try:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][str(i)[:-5] + "_noc"] / tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except KeyError:
                tiling_str[-1][i] =math.ceil(layer[1][str(i)[:-5]][0] / tiling_str[-1][str(i)[:-5] + "_gb"]/ tiling_str[-1][
                                            str(i)[:-5] + "_rf"])
            except:
                raise
    return tiling_str,consumption_dict



def get_score_whole_dnn(tiling_string,consumption,tmp_hw_spec,lp_order_string,input_dnn):
    #check for resource consumption
    [penalty, buffer_not_exceed]=life_eval(consumption, 1, tmp_hw_spec, 0,group_num=1,df_order=lp_order_string)
    if not buffer_not_exceed:
        print('consumption is out of limit')
        return [(penalty[0],penalty[1]), buffer_not_exceed]
    edp_raw=[0,0]
    for layer in range(len(input_dnn)):
        [penalty, buffer_not_exceed] = life_eval(tiling_string[layer], input_dnn[layer][0], tmp_hw_spec,input_dnn[layer][2],group_num=input_dnn[layer][3],df_order=lp_order_string)
        if not buffer_not_exceed:
            print('a oh...')
            return [(penalty[0],penalty[1]), buffer_not_exceed]
        else:
            edp_raw[0]+=penalty[0]
            edp_raw[1]+=penalty[1]
    return  (edp_raw[0], edp_raw[1]), True

    # print(life_eval(tiling_string,input_dnn[0][0],tmp_hw_spec,df_order=lp_order_string))

def resource_allocator_depth_std(input_dnn,tmp_hw_spec):
    input_dnn=copy.deepcopy(input_dnn)
    input_dnn_std=[]
    input_dnn_dw=[]
    for layer in input_dnn:
        if layer[2]==1:
            input_dnn_dw.append(layer)
        else:
            input_dnn_std.append(layer)
    para_dw=0
    para_std=0
    comp_dw=0
    comp_std=0
    for layer in input_dnn:
        if layer[2]==0 or layer[2]==2:
            para_std+=  ((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_in'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])*layer[3])
            comp_std+=((layer[1]['ch_in'][0]*layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])*layer[3])
        elif layer[2]==1:
            para_dw+=(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]+layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0]+layer[0]*layer[1]['ch_out'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])
            comp_dw+=(layer[1]['ch_out'][0]*layer[1]['col_kernel'][0]*\
                        layer[1]['row_kernel'][0]*layer[1]['col_out'][0]*\
                        layer[1]['row_out'][0])

    tmp_hw_spec1 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol']*para_std/(para_dw+para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']*para_std/(para_dw+para_std)), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe']*comp_std/(comp_dw+comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf']*comp_std/(comp_dw+comp_std))
    }
    tmp_hw_spec2 = { \
        'gb_vol': math.ceil(tmp_hw_spec['gb_vol']*para_dw/(para_dw+para_std)), \
        'rf_vol': math.ceil(tmp_hw_spec['rf_vol']*para_dw/(para_dw+para_std)), \
        'num_pe': math.ceil(tmp_hw_spec['num_pe']*comp_dw/(comp_dw+comp_std)), \
        'num_rf': math.ceil(tmp_hw_spec['num_rf']*comp_dw/(comp_dw+comp_std))
    }

    return tmp_hw_spec1, tmp_hw_spec2

# [tiling_choices_dict,tiling_space_rf,tiling_space_gb,pe_array_dim_choices_dict,pe_array_dim_space_all]=tiling_generator(input_dnn,tmp_hw_spec)
#
# print(tiling_choices_dict)
# print(tiling_space_rf)
# print(tiling_space_gb)
# print(pe_array_dim_choices_dict)
# print(pe_array_dim_space_all)
# exit()
# print(pe_array_dim_choices_dict[0])
# [tiling_str,consumption]=tiling_translation([4,2,1,1,2,2,0],[4,2,1,1,2,2,0],[1,1,2,2],0,tiling_choices_dict,pe_array_dim_choices_dict,input_dnn)
# print(consumption)
# exit()

#generate the design space of all possible tiling factors
#the space is partitioned according to alloc_slots based on the rf_noc_template choice (PE array)


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


def adjust_temp(args, _epoch, decay=0.9, freq=1000):
    return args.temp * decay ** (_epoch//freq)


def build_arch_parameters(num_pe_array=4, num_loop_order=7, noc_tiling_dim=4, gb_tiling_dim=6):
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

    gamma_noc_list = []
    for i in range(noc_tiling_dim):
        gamma = nn.Parameter(Variable(1e-3*torch.ones(10).cuda(), requires_grad=True))
        gamma_noc_list.append(gamma)
        params.append(gamma)

    gamma_gb_list = []
    for i in range(gb_tiling_dim):
        gamma = nn.Parameter(Variable(1e-3*torch.ones(10).cuda(), requires_grad=True))
        gamma_gb_list.append(gamma)
        params.append(gamma)


    return params, alpha, beta_dram_list, beta_gb_list, gamma_noc_list, gamma_gb_list


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':20*1024*1024*8, \
    'rf_vol':8000, \
    'num_pe':900, \
    'num_rf':900
}

num_pe_array=4

num_loop_order_std=7
noc_tiling_dim_std=4
gb_tiling_dim_std=6

num_loop_order_dw=6
noc_tiling_dim_dw=3
gb_tiling_dim_dw=5


def search_opt_hw_diff(input_dnn_list):
    opt_hw_list = []

    for input_dnn in input_dnn_list:
        [tmp_hw_spec1, tmp_hw_spec2]=resource_allocator_depth_std(input_dnn, tmp_hw_spec)
        print(tmp_hw_spec1)
        print(tmp_hw_spec2)

        params_std, alpha_std, beta_dram_list_std, beta_gb_list_std, gamma_noc_list_std, gamma_gb_list_std = build_arch_parameters(num_pe_array=4, num_loop_order=num_loop_order_std, noc_tiling_dim=noc_tiling_dim_std, gb_tiling_dim=gb_tiling_dim_std)
        params_dw, alpha_dw, beta_dram_list_dw, beta_gb_list_dw, gamma_noc_list_dw, gamma_gb_list_dw = build_arch_parameters(num_pe_array=4, num_loop_order=num_loop_order_dw, noc_tiling_dim=noc_tiling_dim_dw, gb_tiling_dim=gb_tiling_dim_dw)

        best_metric = 1e8
        best_epoch = 0
        num_exceed = 0
        tbar = tqdm(range(args.epoch), ncols=100)

        params = params_std
        params.extend(params_dw)

        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
        # lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000, 4000, 6000, 8000], gamma=0.5)
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.1)

        input_dnn_dw=[]
        input_dnn_std=[]

        for layer in copy.deepcopy(input_dnn):
            if layer[2] == 1:
                input_dnn_dw.append(layer)
            else:
                input_dnn_std.append(layer)

        for _epoch in tbar:
            if args.adjust_temp:
                temp = adjust_temp(args, _epoch, decay=args.temp_decay, freq=args.temp_freq)
            else:
                temp = args.temp

            params_diff = []
            params_distrib = []

            alpha_sample, alpha_distrib = gumbel_softmax(alpha_std, temperature=temp)
            pe_array_std = alpha_sample.argmax()
            params_diff.append(alpha_sample[pe_array_std])
            params_distrib.append(alpha_distrib)

            alpha_sample, alpha_distrib = gumbel_softmax(alpha_dw, temperature=temp)
            pe_array_dw = alpha_sample.argmax()
            params_diff.append(alpha_sample[pe_array_dw])
            params_distrib.append(alpha_distrib)

            input_lp_order_dram_std = []
            for i in range(num_loop_order_std):
                beta_sample, beta_distrib = gumbel_softmax(beta_dram_list_std[i], temperature=temp)
                order_list = beta_sample.argsort(descending=True)

                for order in order_list:
                    if order not in input_lp_order_dram_std:
                        break

                input_lp_order_dram_std.append(order)
                params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                params_distrib.append(beta_distrib)

            input_lp_order_gb_std = []
            for i in range(num_loop_order_std):
                beta_sample, beta_distrib = gumbel_softmax(beta_gb_list_std[i], temperature=temp)
                order_list = beta_sample.argsort(descending=True)

                for order in order_list:
                    if order not in input_lp_order_gb_std:
                        break

                input_lp_order_gb_std.append(order)
                params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                params_distrib.append(beta_distrib)


            input_lp_order_dram_dw = []
            for i in range(num_loop_order_dw):
                beta_sample, beta_distrib = gumbel_softmax(beta_dram_list_dw[i], temperature=temp)
                order_list = beta_sample.argsort(descending=True)

                for order in order_list:
                    if order not in input_lp_order_dram_dw:
                        break

                input_lp_order_dram_dw.append(order)
                params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                params_distrib.append(beta_distrib)

            input_lp_order_gb_dw = []
            for i in range(num_loop_order_dw):
                beta_sample, beta_distrib = gumbel_softmax(beta_gb_list_dw[i], temperature=temp)
                order_list = beta_sample.argsort(descending=True)

                for order in order_list:
                    if order not in input_lp_order_gb_dw:
                        break

                input_lp_order_gb_dw.append(order)
                params_diff.append((1-beta_distrib[order]).detach() + beta_distrib[order])
                params_distrib.append(beta_distrib)


            #translate the lp_order to string format
            lp_order_string_std = dram_invariant_looporder(pe_array_std,input_lp_order_dram_std, input_lp_order_gb_std)
            lp_order_string_dw = dram_invariant_looporder_dw(pe_array_dw,input_lp_order_dram_dw, input_lp_order_gb_dw)


            # tiling_noc_std=[]
            # if pe_array_std==3:
            #     for i in range(3):
            #         tiling_noc_std.append(randint(1,10))
            # else:
            #     for i in range(4):
            #         tiling_noc_std.append(randint(1,10))
            # #dw
            # tiling_noc_dw=[]
            # for i in range(3):
            #     tiling_noc_dw.append(randint(1,10))
            # #dw -- std
            # tiling_gb_std= []
            # tiling_gb_dw=[]
            # for i in range(6):
            #     tiling_gb_std.append(randint(1,10))
            # for i in range(5):
            #     tiling_gb_dw.append(randint(1,10))


            tiling_noc_std=[]
            if pe_array_std==3:
                for i in range(3):
                    gamma_sample, gamma_distrib = gumbel_softmax(gamma_noc_list_std[i], temperature=temp)
                    buffer_ratio = gamma_sample.argmax()

                    tiling_noc_std.append(buffer_ratio + 1)

                    params_diff.append(gamma_sample[buffer_ratio])
                    params_distrib.append(gamma_distrib)
            else:
                for i in range(4):
                    gamma_sample, gamma_distrib = gumbel_softmax(gamma_noc_list_std[i], temperature=temp)
                    buffer_ratio = gamma_sample.argmax()

                    tiling_noc_std.append(buffer_ratio + 1)

                    params_diff.append(gamma_sample[buffer_ratio])
                    params_distrib.append(gamma_distrib)


            tiling_gb_std=[]
            for i in range(6):
                gamma_sample, gamma_distrib = gumbel_softmax(gamma_gb_list_std[i], temperature=temp)
                buffer_ratio = gamma_sample.argmax()

                tiling_gb_std.append(buffer_ratio + 1)

                params_diff.append(gamma_sample[buffer_ratio])
                params_distrib.append(gamma_distrib)


            tiling_noc_dw=[]
            for i in range(3):
                gamma_sample, gamma_distrib = gumbel_softmax(gamma_noc_list_dw[i], temperature=temp)
                buffer_ratio = gamma_sample.argmax()

                tiling_noc_dw.append(buffer_ratio + 1)

                params_diff.append(gamma_sample[buffer_ratio])
                params_distrib.append(gamma_distrib)


            tiling_gb_dw=[]
            for i in range(5):
                gamma_sample, gamma_distrib = gumbel_softmax(gamma_gb_list_dw[i], temperature=temp)
                buffer_ratio = gamma_sample.argmax()

                tiling_gb_dw.append(buffer_ratio + 1)

                params_diff.append(gamma_sample[buffer_ratio])
                params_distrib.append(gamma_distrib)


            consumption_std = hardware_translation(tiling_noc_std, tiling_gb_std, pe_array_std, tmp_hw_spec1)
            consumption_dw = hardware_translation_dw(tiling_noc_dw, tiling_gb_dw, pe_array_dw, tmp_hw_spec2)
            [tiling_string_std, consumption_std] = tiling_translation(consumption_std, input_dnn_std)
            [tiling_string_dw, consumption_dw] = tiling_translation_dw(consumption_dw, input_dnn_dw)
            #pass for EDP feedback
            #print(pe_array)
            # print(tiling_string)
            # print(lp_order_string)
            penalty_std=get_score_whole_dnn(tiling_string_std, consumption_std, tmp_hw_spec1, lp_order_string_std, input_dnn_std)
            penalty_dw = get_score_whole_dnn(tiling_string_dw, consumption_dw, tmp_hw_spec2, lp_order_string_dw, input_dnn_dw)

            # print(penalty_std)
            # print(penalty_dw)

            if (not penalty_std[1]) or (not penalty_dw[1]):
                loss = -1
                num_exceed += 1
            
            else:  
                metric = (penalty_std[0][0]+penalty_dw[0][0])*(penalty_std[0][1]+penalty_dw[0][1])

                loss = metric 

                if best_metric > metric:
                    best_metric = metric
                    best_epoch = _epoch

                    best_hw = {'pe_array_std':pe_array_std, 'pe_array_dw':pe_array_dw, 

                            'input_lp_order_dram_std':input_lp_order_dram_std, 'input_lp_order_gb_std':input_lp_order_gb_std,
                            'input_lp_order_dram_dw':input_lp_order_dram_dw, 'input_lp_order_gb_dw':input_lp_order_gb_dw,

                            'tiling_noc_std':tiling_noc_std, 'tiling_gb_std':tiling_gb_std,
                            'tiling_noc_dw':tiling_noc_dw, 'tiling_gb_dw':tiling_gb_dw,

                            'input_dnn_all': input_dnn,

                            'edp': metric}
                                
                for item in params_diff:
                    # loss = loss * ((1-item).detach() + item)
                    loss = loss * item

                optimizer.zero_grad()
                lr_policy.step()

                loss.backward()
                optimizer.step()

                loss = loss.item()

            tbar.set_description("[Epoch %d/%d] Loss: %.3f Best Metric: %.3f Best Epoch: %d Num Exceed: [%d/%d]" % 
                                (_epoch + 1, args.epoch, loss, best_metric, best_epoch, num_exceed, _epoch+1))

        opt_hw_list.append(best_hw)

    return opt_hw_list


if __name__ == '__main__':
    input_dnn=[\
    [1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
    [1,{'ch_out':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},2,2],\
    [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},1,1],\
    [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
    [1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]},0,1],\
    ]

    search_opt_hw_diff([input_dnn])





