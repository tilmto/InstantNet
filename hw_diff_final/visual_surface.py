import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse

def read_json(path):
    with open(path, 'r') as f:
        trajectory = json.load(f)
    return trajectory

parser = argparse.ArgumentParser(description="Visualization")
parser.add_argument('--path', type=str, default='',
                    help='saving path of search trajectory')
parser.add_argument('--step', type=int, default=100,
                    help='step for sampling loop orders')
parser.add_argument('--fix_tiling_factor', action='store_true', default=False,
                    help='whether fix the pe array')
parser.add_argument('--fix_loop_order', action='store_true', default=False,
                    help='whether fix the loop order')
parser.add_argument('--fix_tiling', action='store_true', default=False,
                    help='whether fix the tiling factors')
args = parser.parse_args()

path1 = 'trajectory/all.json'

t1 = read_json(path1)

metric = t1['metric']
loop_order = t1['loop_order']
tiling = t1['tiling_factor']
pe = t1['pe_array']

# print(len(loop_order))
# print(len(tiling))
# print(len(pe))
# input()

x = list(range(16))
y = list(range(int(len(tiling)/len(x))))


X, Y = np.meshgrid(x, y)

# Z = np.sqrt(X ** 2 + Y ** 2)
# print(Z.shape)
# input()

Z = np.reshape(metric[:len(x)*len(y)], (len(y), len(x)))
print(Z.shape)


font_big = 20
font_mid = 14
font_small = 12

# fig, ax = plt.subplots(2, 3, figsize=(10,8))
# plt.subplots_adjust(wspace=0.2, hspace=0.35)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))

ax.set_title('EDP - Design Spaced', fontsize=font_big)
ax.set_xlabel('Dataflow * Loop Order', fontsize=font_mid)
ax.set_ylabel('Tiling Factors', fontsize=font_mid)
ax.set_zlabel('EDP', fontsize=font_mid)


plt.savefig('surface.png', bbox_inches='tight')
plt.show()

