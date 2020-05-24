import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
import argparse


if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = 'bn_output.npy'

bn_output = np.load(path, allow_pickle=True).item()

bn_output_before = bn_output['before']
bn_output_after = bn_output['after']

x = range(len(bn_output_before[0])//2)

# print(bn_output_before[0])
# input()
# print(bn_output_before[1])
# input()
# print(bn_output_before[2])
# input()

font_big = 25
font_mid = 20
font_small_y = 20
font_small_x = 20
font_legend = 13
bar_width = 0.14
axis_width = 3


fig, ax = plt.subplots(1, 2, figsize=(20,10))
plt.subplots_adjust(wspace=0.5, hspace=0.35)

for i in range(len(bn_output_before)):
    ax[0].plot(x, bn_output_before[i][::2])
ax[0].set_title('MSE Loss with FP of BN Output (Before Calibration)', fontsize=font_big, fontweight='bold')
ax[0].set_xlabel('Layer', fontsize=font_mid, fontweight='bold')
ax[0].set_ylabel('MSE', fontsize=font_mid, fontweight='bold')
ax[0].legend(['4-bit', '8-bit', '12-bit', '16-bit', '32-bit'], fontsize=font_legend)
ax[0].grid()
# leg = ax[0].legend(fontsize=font_legend)
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

ax[0].xaxis.set_tick_params(labelsize=font_small_x)
ax[0].yaxis.set_tick_params(labelsize=font_small_y)

ax[0].spines['bottom'].set_linewidth(axis_width)
ax[0].spines['left'].set_linewidth(axis_width)
ax[0].spines['top'].set_linewidth(axis_width)
ax[0].spines['right'].set_linewidth(axis_width)


for i in range(len(bn_output_after)):
    ax[1].plot(x, bn_output_after[i][::2])
ax[1].set_title('MSE Loss with FP of BN Output (Before Calibration)', fontsize=font_big, fontweight='bold')
ax[1].set_xlabel('Layer', fontsize=font_mid, fontweight='bold')
ax[1].set_ylabel('MSE', fontsize=font_mid, fontweight='bold')
ax[1].grid()
ax[1].legend(['4-bit', '8-bit', '12-bit', '16-bit', '32-bit'], fontsize=font_legend)
# leg = ax[0].legend(fontsize=font_legend)
# leg.get_frame().set_edgecolor("black")
# leg.get_frame().set_linewidth(2)

ax[1].xaxis.set_tick_params(labelsize=font_small_x)
ax[1].yaxis.set_tick_params(labelsize=font_small_y)

ax[1].spines['bottom'].set_linewidth(axis_width)
ax[1].spines['left'].set_linewidth(axis_width)
ax[1].spines['top'].set_linewidth(axis_width)
ax[1].spines['right'].set_linewidth(axis_width)


plt.savefig('bn_output.png', bbox_inches='tight')
# plt.show()



 