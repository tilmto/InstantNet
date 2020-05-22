import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
x1 = list(range(10))
x2 = list(range(10))
data1 = [73.428, 72.933, 73.428, 72.953, 73.695, 73.754, 72.913, 73.497, 73.586, 73.022]
data2 = [3566, 1134, 6432, 4461, 8085, 8957, 9314, 9179, 6320, 8555]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Weight of EDP Loss')
ax1.set_ylabel('Acc (%)', color=color)
ax1.plot(x1, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_title('Acc/EDP Evolution with Weight of EDP Loss on CIFAR-100')
ax1.grid()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('EDP', color=color)  # we already handled the x-label with ax1
ax2.plot(x2, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xticklabels(['1e-3','1e-4','1e-5','1e-6','1e-7','1e-8','1e-9','1e-10','1e-11','1e-12'])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('edp_weight_ablation.jpg', tight=True)
plt.show()