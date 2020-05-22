import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def get_pareto(acc, edp):
    acc_pareto = []
    edp_pareto = []

    assert len(acc) == len(edp)

    for i in range(len(acc)):
        pareto_flag = True
        for j in range(len(acc)):
            if acc[j] > acc[i] and edp[j] < edp[i]:
                pareto_flag = False
                break
        if pareto_flag:
            acc_pareto.append(acc[i])
            edp_pareto.append(edp[i])

    arg = np.array(edp_pareto).argsort()

    return np.array(acc_pareto)[arg], np.array(edp_pareto)[arg]


acc = [75.633, 74.031, 74.911, 76.384, 75.86, 75.79, 75.48, 74.96, 75.34, 74.79, 74.96, 74.91, 75.31]
edp = [363.54, 204.78, 313.38, 367.89, 226.09, 201.01, 277.71, 156.28, 196.8, 177.37, 249.42, 232.09, 165.5]

acc_pareto, edp_pareto = get_pareto(acc, edp)

fig, ax = plt.subplots(1,2)

ax[0].set_xlabel('EDP')
ax[0].set_ylabel('Acc (%)')
ax[0].plot(edp, acc, '*r')
ax[0].plot(edp_pareto, acc_pareto, '.-y')
ax[0].legend(['Searched Arch', 'Pareto Optimal'])
ax[0].set_title('Acc - EDP Distrib on CIFAR-100')
ax[0].grid()


acc = [94.41, 94.3, 94.46, 94.65, 94.77, 94.45, 94.52, 94.4, 94.62, 94.57, 94.62, 95.16, 95.06, 94.88]
edp = [145.06, 155.63, 177.37, 249.42, 232.09, 169.04, 214.48, 215.19, 165.5, 187.81, 141.07, 437.44, 422.3, 223.45]

acc_pareto, edp_pareto = get_pareto(acc, edp)

ax[1].set_xlabel('EDP')
ax[1].set_ylabel('Acc (%)')
ax[1].plot(edp, acc, '*r')
ax[1].plot(edp_pareto, acc_pareto, '.-y')
ax[1].legend(['Searched Arch', 'Pareto Optimal'])
ax[1].set_title('Acc - EDP Distrib on CIFAR-10')
ax[1].grid()


plt.savefig('acc_edp_tradeoff.jpg', tight=True)
plt.show()
