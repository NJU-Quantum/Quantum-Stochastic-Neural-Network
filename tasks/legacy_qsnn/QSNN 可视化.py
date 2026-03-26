import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置matplotlib支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 演示：矩阵A很稀疏，不代表e^A也稀疏
# 原因：e^A = I + A + A^2/2! + A^3/3! + ...
# A^2, A^3等往往会产生填充项（fill-in）
# 即使H是稀疏的，如最近邻耦合的三对角矩阵，U(t)=e^{-itH}也会变得稠密

def demonstrate_matrix_exponential_sparsity():
    # 创建一个稀疏的三对角矩阵A（模拟最近邻耦合的哈密顿量）
    n = 10  # 矩阵大小
    # 用numpy创建三对角矩阵
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2  # 主对角线
        if i > 0:
            A[i, i-1] = -1  # 下对角线
        if i < n-1:
            A[i, i+1] = -1  # 上对角线

    # 转换为torch张量并计算矩阵指数 e^A
    A_torch = torch.from_numpy(A).float()
    exp_A_torch = torch.matrix_exp(A_torch)
    exp_A = exp_A_torch.numpy()

    # 计算稀疏度：非零元素比例
    sparsity_A = np.count_nonzero(A) / A.size
    sparsity_exp_A = np.count_nonzero(exp_A) / exp_A.size

    print(f"原矩阵A的稀疏度（非零元素比例）：{sparsity_A:.3f}")
    print(f"e^A的稀疏度（非零元素比例）：{sparsity_exp_A:.3f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 原始矩阵A
    axes[0].imshow(A, cmap='viridis', aspect='equal')
    axes[0].set_title(f'原始矩阵A (稀疏度: {sparsity_A:.3f})')
    axes[0].set_xlabel('列')
    axes[0].set_ylabel('行')

    # e^A
    im = axes[1].imshow(exp_A, cmap='viridis', aspect='equal')
    axes[1].set_title(f'e^A (稀疏度: {sparsity_exp_A:.3f})')
    axes[1].set_xlabel('列')
    axes[1].set_ylabel('行')

    # 添加颜色条
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    # 移除tight_layout以避免警告
    plt.show()

# 调用函数
if __name__ == "__main__":
    demonstrate_matrix_exponential_sparsity()

