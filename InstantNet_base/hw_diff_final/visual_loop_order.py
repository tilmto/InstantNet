import numpy as np 
import matplotlib.pyplot as plt 
import json
import argparse

def read_json(path):
    with open(path, 'r') as f:
        trajectory = json.load(f)
    return trajectory

parser = argparse.ArgumentParser(description="Visualization")
parser.add_argument('--path', type=str, default='',
                    help='saving path of search trajectory')
parser.add_argument('--step', type=int, default=1,
                    help='step for sampling loop orders')
parser.add_argument('--fix_pe_array', action='store_true', default=False,
                    help='whether fix the pe array')
parser.add_argument('--fix_loop_order', action='store_true', default=False,
                    help='whether fix the loop order')
parser.add_argument('--fix_tiling', action='store_true', default=False,
                    help='whether fix the tiling factors')
args = parser.parse_args()

path1 = 'trajectory/loop_order1.json'
path2 = 'trajectory/loop_order2.json'
path3 = 'trajectory/loop_order3.json'
path4 = 'trajectory/loop_order4.json'
path5 = 'trajectory/loop_order5.json'
# path6 = 'trajectory/pe_array6.json'
# path7 = 'trajectory/pe_array7.json'
# path8 = 'trajectory/pe_array8.json'
# path9 = 'trajectory/pe_array9.json'
# path10 = 'trajectory/pe_array10.json'

t1 = read_json(path1)
t2 = read_json(path2)
t3 = read_json(path3)
t4 = read_json(path4)
t5 = read_json(path5)
# t6 = read_json(path6)
# t7 = read_json(path7)
# t8 = read_json(path8)
# t9 = read_json(path9)
# t10 = read_json(path10)

m1 = t1['metric'][::args.step]
m2 = t2['metric'][::args.step]
m3 = t3['metric'][::args.step]
m4 = t4['metric'][::args.step]
m5 = t5['metric'][::args.step]
# m6 = t6['metric']
# m7 = t7['metric']
# m8 = t8['metric']
# m9 = t9['metric']
# m10 = t10['metric']

x1 = list(range(len(m1)))
x2 = list(range(len(m2)))
x3 = list(range(len(m3)))
x4 = list(range(len(m4)))
x5 = list(range(len(m5)))
# x6 = list(range(len(m6)))
# x7 = list(range(len(m7)))
# x8 = list(range(len(m8)))
# x9 = list(range(len(m9)))
# x10 = list(range(len(m10)))


font_big = 20
font_mid = 14
font_small = 12

# fig, ax = plt.subplots(2, 3, figsize=(10,8))
# plt.subplots_adjust(wspace=0.2, hspace=0.35)

plt.plot(x1, m1, '-')
plt.plot(x2, m2, '-')
plt.plot(x3, m3, '-')
plt.plot(x4, m4, '-')
plt.plot(x5, m5, '-')
# plt.plot(x6, m6, '-^')
# plt.plot(x7, m7, '-^')
# plt.plot(x8, m8, '-^')
# plt.plot(x9, m9, '-^')
# plt.plot(x10, m10, '-^')

# plt.ylim((0,100))

plt.title('EDP - Loop Order', fontsize=font_big)
plt.ylabel('EDP', fontsize=font_mid)
plt.xlabel('Loop Order Options', fontsize=font_mid)
# plt.legend(['random_seed1','random_seed2','random_seed3','random_seed4','random_seed5',
#             'random_seed6','random_seed7','random_seed8','random_seed9','random_seed10'], fontsize=font_mid)
plt.grid()

# ax[0,1].plot(cc_resnet38_static_cifar100, acc_resnet38_static_cifar100, '-^')
# ax[0,1].plot(cc_resnet38_dp_cifar100, acc_resnet38_dp_cifar100, '-^')
# ax[0,1].set_title('ResNet-38@CIFAR-100', fontsize=font_big)
# ax[0,1].set_ylabel('Accuracy(%)', fontsize=font_mid)
# ax[0,1].set_xlabel('MACs', fontsize=font_mid)
# ax[0,1].legend(['static','dynamic'], fontsize=font_mid)
# ax[0,1].grid()
# ax[0,1].xaxis.set_tick_params(labelsize=font_small)
# ax[0,1].yaxis.set_tick_params(labelsize=font_small)


plt.savefig('loop_order2.png', bbox_inches='tight')
plt.show()

