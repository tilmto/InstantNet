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
import copy
import json

num_pe_array = 4
num_loop_order = 7
num_pe_array_dim = 10
num_order = 2
layer = 0

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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


def permute(orig_list, permutation_list, start_index=0):
    if start_index == len(orig_list)-1:
        permutation_list.append(orig_list)
        # print(orig_list)
    else:
        for i in range(start_index, len(orig_list)):
            new_list = copy.deepcopy(orig_list)
            tmp = new_list[i]
            new_list[i] = new_list[start_index]
            new_list[start_index] = tmp
            permute(new_list, permutation_list, start_index+1)



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
parser.add_argument('--step', type=int, default=1,
                    help='step when exploring tiling factor')
parser.add_argument('--random_seed', type=int, default=10,
                    help='random seed for np.random')
parser.add_argument('--save_path', type=str, default='',
                    help='saving path of search trajectory')
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
args = parser.parse_args()

tiling1 = fpga_tiling_generator(input_dnn,tmp_hw_spec)

if args.fix_pe_array:
    pe_array_exhaust = [get_fixed_pe_array(num_pe_array, args.random_seed)]
else:
    pe_array_exhaust = list(range(num_pe_array))

if args.fix_loop_order_dram:
    input_lp_order_dram_exhaust = [get_fixed_loop_order(num_loop_order, args.random_seed)]
else:
    tmp_list_exhaust = []
    orig_list = list(np.arange(num_loop_order))
    permute(orig_list, tmp_list_exhaust)
    input_lp_order_dram_exhaust = tmp_list_exhaust

if args.fix_loop_order_gb:
    input_lp_order_gb_exhaust = [get_fixed_loop_order(num_loop_order, args.random_seed)]
else:
    if not args.fix_loop_order_dram:
        input_lp_order_gb_exhaust = tmp_list_exhaust
    else:
        tmp_list_exhaust = []
        orig_list = list(np.arange(num_loop_order))
        permute(orig_list, tmp_list_exhaust)
        input_lp_order_gb_exhaust = tmp_list_exhaust

if args.fix_pe_array_dim:
    pe_array_dim_choices_exhaust = [get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)]
else:
    pe_array_dim_choices_exhaust = list(range(num_pe_array_dim))


pe_array_fix = get_fixed_pe_array(num_pe_array, args.random_seed)
pe_array_dim_choices_fix = get_fixed_pe_array_dim(num_pe_array_dim, args.random_seed)
tiling_space_1 = tiling1.tiling_space_partition(pe_array_fix, layer, pe_array_dim_choices_fix)
if args.fix_tiling_factor:
    tiling_choices_exhaust = [get_fixed_tiling_factor(tiling_space_1, args.random_seed)]
else:
    tiling_choices_exhaust = []
    for pe_array in range(num_pe_array):
        sub_list = []
        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim) 
            
            tiling_scheme = []
            for t1 in range(tiling_space_1[0]):
                for t2 in range(tiling_space_1[1]):
                    for t3 in range(tiling_space_1[2]):
                        for t4 in range(tiling_space_1[3]):
                            for t5 in range(tiling_space_1[4]):
                                if pe_array == 0:
                                    sub_sub_list.append([t1,t2,t3,t4,t5])
                                else:
                                    for t6 in range(tiling_space_1[5]):
                                        for t7 in range(tiling_space_1[6]):
                                            sub_sub_list.append([t1,t2,t3,t4,t5,t6,t7])
            sub_list.append(sub_sub_list)

        tiling_choices_exhaust.append(sub_list)


if args.fix_tiling_order:
    tiling_choices_order_exhaust = [get_fixed_tiling_order(tiling_space_1, args.random_seed)]
else:
    tiling_choices_order_exhaust = []
    for pe_array in range(num_pe_array):
        sub_list = []
        for pe_array_dim in range(num_pe_array_dim):
            sub_sub_list = []
            tiling_space_1 = tiling1.tiling_space_partition(pe_array,layer,pe_array_dim) 
            
            tiling_scheme = []
            for t1 in [0, 1]:
                for t2 in [0, 1]:
                    for t3 in [0, 1]:
                        for t4 in [0, 1]:
                            for t5 in [0, 1]:
                                if pe_array == 0:
                                    sub_sub_list.append([t1,t2,t3,t4,t5])
                                else:
                                    for t6 in [0, 1]:
                                        for t7 in [0, 1]:
                                            sub_sub_list.append([t1,t2,t3,t4,t5,t6,t7])
            sub_list.append(sub_sub_list)

        tiling_choices_order_exhaust.append(sub_list)


############################
#user interface
############################

_epoch = 0
num_exceed = 0
best_metric = 1e8
trajectory = {'pe_array':[], 'loop_order_dram':[], 'loop_order_gb':[], 'pe_array_dim':[], 
             'tiling_factor':[], 'tiling_order':[], 'metric':[]}


tbar1 = tqdm(pe_array_exhaust, ncols=80)

for pe_array in tbar1:
    tbar2 = tqdm(input_lp_order_dram_exhaust[::args.step], ncols=80)

    for input_lp_order_dram in tbar2:
        tbar3 = tqdm(input_lp_order_gb_exhaust[::args.step], ncols=80)

        for input_lp_order_gb in tbar3:
            lp_order_string = dram_invariant_looporder(pe_array,input_lp_order_dram, input_lp_order_gb)

            tbar4 = tqdm(pe_array_dim_choices_exhaust, ncols=80)

            for pe_array_dim_choices in tbar4:
                if args.fix_tiling_factor:
                    tbar5 = tqdm(tiling_choices_exhaust, ncols=80)
                else:
                    tbar5 = tqdm(tiling_choices_exhaust[pe_array][pe_array_dim_choices], ncols=80)

                for tiling_choices in tbar5:
                    if args.fix_tiling_order:
                        tbar6 = tqdm(tiling_choices_order_exhaust, ncols=80)
                    else:
                        tbar6 = tqdm(tiling_choices_order_exhaust[pe_array][pe_array_dim_choices], ncols=80)

                    for tiling_choices_order in tbar6:
                        tiling_string = tiling1.tiling_translation(layer,pe_array,pe_array_dim_choices,tiling_choices,tiling_choices_order)
                        metric, valid = life_eval(tiling_string, 1, tmp_hw_spec, df_order=lp_order_string)
                        _epoch += 1

                        if valid:
                            trajectory['pe_array'].append(pe_array)
                            trajectory['loop_order_dram'].append(input_lp_order_dram)
                            trajectory['loop_order_gb'].append(input_lp_order_gb)
                            trajectory['pe_array_dim'].append(pe_array_dim_choices)
                            trajectory['tiling_factor'].append(tiling_choices)
                            trajectory['tiling_order'].append(tiling_choices_order)
                            trajectory['metric'].append(metric)

                            if metric < best_metric:
                                best_metric = metric

                        else:
                            num_exceed += 1
                            print('DSP limit exceeded')
                            continue 

                        tbar6.set_description("Metric: %.3f Best Metric: %.3f Num Exceed: [%d/%d]" % (metric, best_metric, num_exceed, _epoch))


if args.save_path:
    with open(args.save_path, 'w') as f:
        json.dump(trajectory, f, cls=NpEncoder)

print("Exhaustive Sample Best Metric:", best_metric)





