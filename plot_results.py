import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np

COLORS = [
    '#ff9b00',  # orange
    '#2a6494',  # blue
    '#595958',  # gray
    '#ff0000',
    '#00ff00',
    '#0000ff'
]

xdata = list(range(20))
# print(convention)
# print(xdata)

# ydata_mean = np.array([56.19, 73.41, 73.94, 65.13, 92.88, 88.47, 86.85, 91.73, 99.94, 114.94, 114.76])
# ydata_std = np.array([3.01, 3.64, 3.63, 3.41, 3.95, 3.97, 3.40, 4.39, 4.18, 3.10, 3.51])

# plt.clf()
# # plt.title(f"{name} Convention {convention}", fontsize=20)
# plt.xlabel("Episode", color="black", fontsize=20)
# plt.ylabel("Episode Reward", color="black", fontsize=20)

# plt.plot(xdata,
#          ydata_mean,
#          linewidth=3.0,
#          color=COLORS[0]
#          )
# plt.fill_between(xdata,
#                  ydata_mean - ydata_std,
#                  ydata_mean + ydata_std,
#                  alpha = 0.2,
#                  color=COLORS[0])

# for pos in ['right', 'top']:
#     plt.gca().spines[pos].set_visible(False)
#     plt.gca().spines['bottom'].set_color('gray')
#     plt.gca().spines['left'].set_color('gray')
    
# plt.gca().tick_params(axis='x', colors='black')
# plt.gca().tick_params(axis='y', colors='black')

# plt.yticks(fontsize=15)
# plt.xticks(fontsize=15)
# # plt.gca().set_xticks([0, 10])
# # plt.gca().set_yticks([50, 100])
# plt.grid(visible=True, which='major', alpha=0.5)

# plt.savefig("few_shot_plot.png")
# plt.show()

# ydatas = [np.load('few_shot_results_old.npy'), np.load('few_shot_results_simple.npy'), np.load('few_shot_results_baseline.npy')]

# ydatas = [np.load('few_shot_results7.npy'), np.load('few_shot_results4.npy'), np.load('few_shot_results8.npy')]
ydatas = [np.load('few_shot_results7.npy'), np.load('few_shot_results_simple7.npy'), np.load('few_shot_results4.npy'), np.load('few_shot_results_simple4.npy'), np.load('few_shot_results2.npy'), np.load('few_shot_results_simple2.npy')]

use_mean = True



plt.clf()
# plt.title(f"{name} Convention {convention}", fontsize=20)
plt.xlabel("Episode", color="gray", fontsize=12)
plt.ylabel("Episode Reward", color="gray", fontsize=12)

for i in range(len(ydatas)):
    ydata = ydatas[i]
    
    print(ydata)

    ydata = np.array(ydata).transpose()
    if use_mean:

        ydata_mean = np.mean(ydata, axis=1)

        print(ydata.shape)
        ydata_std = np.std(ydata, axis=1) / np.sqrt(np.shape(ydata)[1])

        ydata_lower = ydata_mean - ydata_std
        ydata_higher = ydata_mean + ydata_std
    else:
        ydata_mean = np.median(ydata, axis=1)

        print(ydata.shape)

        ydata_lower = np.percentile(ydata, 25, axis=1)
        ydata_higher = np.percentile(ydata, 75, axis=1)
    plt.plot(xdata,
             ydata_mean,
             linewidth=3.0,
             color=COLORS[i // 2],
             linestyle='--' if i % 2 == 1 else '-'
             )
    plt.fill_between(xdata,
                     ydata_lower,
                     ydata_higher,
                     alpha = 0.2,
                     color=COLORS[i // 2])

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_color('gray')
    
plt.gca().tick_params(axis='x', colors='gray')
plt.gca().tick_params(axis='y', colors='gray')

plt.yticks(fontsize=15)
plt.xticks(range(0, 20, 2), fontsize=15)
# plt.gca().set_xticks([0, 10])
# plt.gca().set_yticks([50, 100])
plt.grid(visible=True, which='major', alpha=0.5)

plt.savefig("few_shot_plot.png")
plt.show()
