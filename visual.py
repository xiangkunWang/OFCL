# import matplotlib.pyplot as plt
# import numpy as np
# dis_open = [64.58, 67.83, 68.39, 66.88, 67.52, 68.84]
# AUROC = [62.59, 62.37, 61.63, 62.48, 62.16, 60.74]
# FPR = [72.32, 73.70, 74.73, 73.40, 73.56, 75.92]
# x = range(6)
# fig = plt.figure(figsize=(6, 5.6))
# plt.xticks(range(6), [5, 10, 15, 20, 25, 30])
# plt.plot(x, dis_open, marker='o', markersize=12, label='$ACC_N$', markeredgecolor="white",linewidth=3)
# # plt.fill_between(x, np.subtract(data1, std1), np.add(data1, std1), alpha=0.3)
# plt.plot(x, AUROC, marker='^', markersize=14, label='$AUC_N$', markeredgecolor="white",linewidth=3)
# # plt.fill_between(x, np.subtract(data2, std2), np.add(data2, std2), alpha=0.3)
# plt.plot(x, FPR, marker='v', markersize=14, label='$FPR_N$', markeredgecolor="white",linewidth=3)
# # plt.fill_between(x, np.subtract(data3, std3), np.add(data3, std3), alpha=0.3)
# plt.xlabel('the # of prompts of each  session', fontsize=22)
# # plt.ylabel('$ACC_N$/$AUC_N$/$FPR_N$', fontsize=22)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.xlim(-0.3, 5.3)
# plt.ylim(55, 80)
# plt.grid(linestyle='dashed', linewidth=1.2)
# plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, fontsize=16)  # 添加图例
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

data = [[56.93, 67.89, 68.32, 69.39, 67.01],
        [64.66, 67.52, 67.84, 68.45, 68.26],
        [67.90, 69.76, 68.08, 67.47,65.36],
        [67.32, 68.97, 63.30, 62.67, 61.29],
        [67.50, 67.68, 66.35, 60.66, 60.62]]
ticks = [1, 5, 10, 20, 25]
num_ticks = len(ticks)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(data, cmap='Greens', interpolation='nearest', vmin=50, vmax=70)
# 设置新的坐标位置
ax.set_xticks(np.linspace(0, len(data[0])-1, num_ticks))
ax.set_yticks(np.linspace(0, len(data)-1, num_ticks))
# 设置显示的坐标标签
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
# 翻转y轴方向
ax.invert_yaxis()
# 添加数值标签
for i in range(len(data)):
    for j in range(len(data[i])):
        text = ax.text(j, i, f'{data[i][j]}', ha='center', va='center', color='w',fontsize=15)
# 添加颜色条
# cbar = ax.figure.colorbar(im)
ax.set_xlabel('Selected size',fontsize=22)
ax.set_ylabel('Prompt length',fontsize=22)
plt.setp(ax.get_xticklabels(), fontsize=20)  # 设置坐标轴标签字体大小为8
plt.setp(ax.get_yticklabels(), fontsize=20)
# 显示图像
plt.show()