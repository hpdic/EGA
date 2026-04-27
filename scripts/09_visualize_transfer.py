# Copyright (c) 2026 Dongfang Zhao <dzhao@uw.edu>
# Created: April 27, 2026

import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小，确保清晰度
plt.rcParams.update({"font.size": 16, "font.family": "serif"})

# 准备数据
k_labels = ["Recall@1", "Recall@3", "Recall@5", "Recall@10"]
nprobes = [1, 5, 10]

# 实验数据整理
data = {
    "Original CLIP": [
        [0.6170, 0.9555, 0.9925],
        [0.5887, 0.9442, 0.9873],
        [0.5707, 0.9399, 0.9857],
        [0.5363, 0.9291, 0.9819]
    ],
    "EGA": [
        [0.8580, 0.9985, 0.9995],
        [0.8418, 0.9978, 0.9997],
        [0.8332, 0.9986, 0.9997],
        [0.8034, 0.9981, 0.9997]
    ],
    "ICon": [
        [0.7875, 0.9910, 0.9990],
        [0.7665, 0.9862, 0.9973],
        [0.7543, 0.9839, 0.9976],
        [0.7349, 0.9809, 0.9966]
    ],
    "SRL": [
        [0.8305, 0.9925, 0.9990],
        [0.7983, 0.9893, 0.9988],
        [0.7902, 0.9873, 0.9981],
        [0.7719, 0.9844, 0.9975]
    ]
}

# 紫金风格配色字典
# EGA 使用洛杉矶湖人队专属紫 #552583
# 其他 baseline 使用金色、银色和深灰色
colors = {
    "EGA": "#552583", 
    "SRL": "#FDB927", 
    "ICon": "#333333", 
    "Original CLIP": "#C0C0C0"
}

markers = {"EGA": "s", "SRL": "D", "ICon": "^", "Original CLIP": "o"}

# 创建 1 行 4 列的画布
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for i in range(4):
    ax = axes[i]
    for model_name, model_data in data.items():
        # 增加线宽和标记点大小，方便观察趋势
        ax.plot(nprobes, model_data[i], label=model_name, 
                color=colors[model_name], marker=markers[model_name], 
                linewidth=3, markersize=12, alpha=0.9)
    
    # 标题和标签
    ax.set_title(k_labels[i], fontsize=20, fontweight="bold", pad=15)
    ax.set_xlabel("nprobe (Search Scope)", fontsize=16)
    
    # 设置 x 轴刻度
    ax.set_xticks(nprobes)
    
    # 调整 y 轴范围，重点展示 0.5 到 1.0 的变化
    # 这能显著放大 nprobe 1 时各模型的差距
    ax.set_ylim(0.5, 1.02)
    
    # 美化网格线
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if i == 0:
        ax.set_ylabel("Recall Score (Generalization)", fontsize=18)
        # 调整图例位置和字体
        ax.legend(loc="lower right", fontsize=14, frameon=True, shadow=True)

# 调整子图间距，防止坐标轴重叠
plt.tight_layout()

# 保存高质量图片
save_path = "cifar10_generalization_v2.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")
plt.show()