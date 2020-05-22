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

from tqdm import tqdm
import argparse
import json


num_pe_array = 4
num_loop_order = 7
num_pe_array_dim = 10
num_order = 2
layer = 0


def dram_variant_looporder(input_lp_order_dram,input_lp_order_sram):
    return None


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


input_dnn=[\
#[1,{'ch_out':[64,0],'ch_in':[3,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[64,0],'ch_in':[64,0],'batch':[1,0],'col_out':[224,0],'row_out':[224,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[64,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[128,0],'ch_in':[128,0],'batch':[1,0],'col_out':[112,0],'row_out':[112,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
[1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[256,0],'ch_in':[256,0],'batch':[1,0],'col_out':[56,0],'row_out':[56,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[256,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[28,0],'row_out':[28,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
#[1,{'ch_out':[512,0],'ch_in':[512,0],'batch':[1,0],'col_out':[14,0],'row_out':[14,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\

# [1,{'ch_out':[256,0],'ch_in':[128,0],'batch':[1,0],'col_out':[8,0],'row_out':[8,0],'row_kernel':[3,0],'col_kernel':[3,0]}],\
]


#fpga dedicated 706
tmp_hw_spec={\
    'gb_vol':2*1024*1024, \
    'rf_vol':512, \
    'num_pe':144, \
    'num_rf':144
}



parser = argparse.ArgumentParser(description="Random Hardware Search")
parser.add_argument('--epoch', type=int, default=10000,
                    help='num of epochs')
parser.add_argument('--trial', type=int, default=1,
                    help='num of trials')
parser.add_argument('--limit', type=float, default=None,
                    help='metric limit')
parser.add_argument('--random_seed', type=int, default=10,
                    help='random seed for np.random')
parser.add_argument('--fix_pe_array', action='store_true', default=False,
                    help='whether to fix the pe array')
parser.add_argument('--fix_pe_array_dim', action='store_true', default=False,
                    help='whether to fix the pe array dim choice')
parser.add_argument('--fix_loop_order_dram', action='store_true', default=False,
                    help='whether to fix the loop order of dram')
parser.add_argument('--fix_loop_order_gb', action='store_true', default=False,
                    help='whether to fix the loop order of global buffer')
parser.add_argument('--fix_tiling_factor', action='store_true', default=False,
                    help='whether fix the tiling factors')
parser.add_argument('--fix_tiling_order', action='store_true', default=False,
                    help='whether fix the tiling orders')
parser.add_argument('--save_path', type=str, default='random_trial_info.json',
                    help='save path of trial info')
args = parser.parse_args()



############################
#user interface
############################

tiling1 = fpga_tiling_generator(input_dnn,tmp_hw_spec)


if args.fix_pe_array:
    pe_array_fix = get_fixed_pe_array(num_pe_array, args.random_seed)
if args.fix_loop_order_dram:
    input_lp_order_dram_fix = get_fixed_loop_order(num_loop_order, args.random_seed)
if args.fix_loop_order_gb:
    input_lp_order_gb_fix = get_fixed_loop_order(num_loop_order, args.random_seed)
if args.fix_pe_array_dim:
    pe_array_dim_choices_fix = get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)

pe_array_fix = get_fixed_pe_array(num_pe_array, args.random_seed)
pe_array_dim_choices_fix = get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)
tiling_space_1 = tiling1.tiling_space_partition(pe_array_fix, layer, pe_array_dim_choices_fix)
if args.fix_tiling_factor:
    tiling_choices_fix = get_fixed_tiling_factor(tiling_space_1, args.random_seed)
if args.fix_tiling_order:
    tiling_choices_order_fix = get_fixed_tiling_order(tiling_space_1, args.random_seed)


stop_epoch_list = []
metric_list = []
best_metric_list = []
best_epoch_list = []

for _trial in range(args.trial):
    best_metric = 1e8
    best_epoch = 0
    num_exceed = 0
    tbar = tqdm(range(args.epoch), ncols=150)

    for _epoch in tbar:
        if args.fix_pe_array:
            pe_array = pe_array_fix
        else:
            pe_array = randint(0,3)

        if args.fix_loop_order_gb:
            input_lp_order_gb = input_lp_order_gb_fix
        else:
            input_lp_order_gb = list(range(7))
            shuffle(input_lp_order_gb)

        if args.fix_loop_order_dram:
            input_lp_order_dram = input_lp_order_dram_fix
        else: 
            input_lp_order_dram = list(range(7))
            shuffle(input_lp_order_dram)

        lp_order_string = dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb)

        if args.fix_pe_array_dim:
            pe_array_dim_choices = pe_array_dim_choices_fix
        else:
            pe_array_dim_choices = randint(0,9)
        
        tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim_choices) 
       
        if args.fix_tiling_factor:
            tiling_choices = tiling_choices_fix
        else:
            tiling_choices = []
            for i in tiling_space_1:
                tiling_choices.append(randint(0,i-1))
        
        if args.fix_tiling_order:
            tiling_choices_order = tiling_choices_order_fix
        else:
            tiling_choices_order=[]
            for i in range(len(tiling_choices)):
                tiling_choices_order.append(randint(0,1))
        
        tiling_string=tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,tiling_choices,tiling_choices_order)

        metric, valid = life_eval(tiling_string,1,tmp_hw_spec,df_order=lp_order_string)

        if valid:            
            if best_metric > metric:
                best_metric = metric
                best_epoch = _epoch
        else:
            metric = -1
            num_exceed += 1

        tbar.set_description("[Trial %d/%d][Epoch %d/%d] Metric: %.3f Best Metric: %.3f Num Exceed: [%d/%d]" % (_trial+1, args.trial, _epoch + 1, args.epoch, metric, best_metric, num_exceed, _epoch+1))

        if args.limit:
            if metric < args.limit:
                print('Epoch:', _epoch, 'Metric:', metric)
                stop_epoch_list.append(_epoch)
                metric_list.append(metric)
                break

            if _epoch >= args.epoch-1:
                print('Epoch:', _epoch, 'Metric:', metric)
                stop_epoch_list.append(_epoch)
                metric_list.append(metric)

    best_metric_list.append(best_metric)
    best_epoch_list.append(best_epoch)


print('Best Metric List', best_metric_list)
print('Average Best Metric', sum(best_metric_list)/len(best_metric_list))

print('Best Epoch List', best_epoch_list)
print('Average Best Epoch', sum(best_epoch_list)/len(best_epoch_list))

trial_info = {'best_metric_list': best_metric_list, 'avg_best_metric': sum(best_metric_list)/len(best_metric_list), 
              'best_epoch_list': best_epoch_list, 'avg_best_epoch': sum(best_epoch_list)/len(best_epoch_list)}

with open(args.save_path, 'w') as f:
    json.dump(trial_info, f)

if args.limit:
    print('Stop Epoch List', stop_epoch_list)
    print('Average Stop Epoch', sum(stop_epoch_list)/len(stop_epoch_list))
    print('Metric List', metric_list)
    print('Average Metric', sum(metric_list)/len(metric_list))






