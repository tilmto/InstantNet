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
parser.add_argument('--fix_tiling_factor', action='store_true', default=False,
                    help='whether fix the pe array')
parser.add_argument('--fix_loop_order', action='store_true', default=False,
                    help='whether fix the loop order')
parser.add_argument('--fix_tiling', action='store_true', default=False,
                    help='whether fix the tiling factors')
args = parser.parse_args()

traj = read_json(args.path)

print(traj.keys())

pe_array = np.array(traj['pe_array'])
loop_order = np.array(traj['loop_order'])
tiling_factor = np.array(traj['tiling_factor'])
metric = np.array(traj['metric'])

print(tiling_factor[metric>10000])
print(metric[metric>10000])

font_big = 20
font_mid = 14
font_small = 12

x = range(len(metric))

plt.figure(figsize=(60,10))
plt.plot(x, metric, '-o')
# plt.ylim((0,100))
plt.title('EDP - Tiling Factor', fontsize=font_big)
plt.ylabel('EDP', fontsize=font_mid)
plt.xlabel('Tiling Factor Options', fontsize=font_mid)
plt.grid()
plt.savefig('analysis.png', bbox_inches='tight')
plt.show()

