import numpy as np
import matplotlib.pyplot as plt

acc = [73.428, 72.933, 73.428, 72.953, 73.695, 73.754, 72.913, 73.497, 73.586, 73.022]
edp = [3785, 1211, 3898, 4461, 8085, 8957, 9314, 9179, 6320, 8555]

baseline = [73.23, 12573.87]  # [acc, edp]

fig, ax1 = plt.subplots()

ax1.set_xlabel('EDP')
ax1.set_ylabel('Acc (%)')
ax1.plot(edp, acc, '*r')
ax1.plot(baseline[1], baseline[0], '^b')
ax1.legend(['NACoS', 'Sequential Search'])
ax1.set_title('Acc - EDP Distrib on CIFAR-100')
ax1.grid()

plt.savefig('acc_edp_tradeoff.jpg', tight=True)
plt.show()