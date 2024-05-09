import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm


def kernel_function(x, y, kernel_width):
    """高斯核函数"""
    sq_dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-sq_dists / (2 * kernel_width ** 2))


def compute_kid(real_samples, generated_samples, kernel_width=1.0, num_subsets=100, subset_size=1000):
    """
    计算KID (Kernel Inception Distance)

    参数:
        real_samples (numpy.ndarray): 真实样本数据，形状为 (N, d)，其中 N 是样本数量，d 是样本维度。
        generated_samples (numpy.ndarray): 生成样本数据，形状与 real_samples 相同。
        kernel_width (float): 高斯核的宽度参数。
        num_subsets (int): 从真实和生成样本中随机选择的子集数量。
        subset_size (int): 每个子集的大小。

    返回:
        float: 计算的KID值。
    """
    n_samples = min(len(real_samples), len(generated_samples))

    # 计算真实和生成样本之间的核矩阵
    K_rr = kernel_function(real_samples[:n_samples], real_samples[:n_samples], kernel_width)
    K_gg = kernel_function(generated_samples[:n_samples], generated_samples[:n_samples], kernel_width)
    K_rg = kernel_function(real_samples[:n_samples], generated_samples[:n_samples], kernel_width)

    # 计算均值嵌入
    m_r = np.mean(K_rr, axis=0)
    m_g = np.mean(K_gg, axis=0)

    # 计算KID估计值
    kid_estimate = 0
    for _ in range(num_subsets):
        # 从真实和生成样本中随机选择子集
        indices = np.random.choice(n_samples, subset_size, replace=False)
        m_r_subset = m_r[indices]
        m_g_subset = m_g[indices]
        K_rr_subset = K_rr[np.ix_(indices, indices)]
        K_gg_subset = K_gg[np.ix_(indices, indices)]
        K_rg_subset = K_rg[np.ix_(indices, indices)]

        # 计算子集上的均值和协方差
        mean_diff = m_r_subset - m_g_subset
        cov_mean_diff = (K_rr_subset + K_gg_subset - K_rg_subset - K_rg_subset.T) / (2 * subset_size)

        # 计算当前子集的KID值并累加到总估计值中
        kid_estimate += np.sqrt(mean_diff.T @ sqrtm(cov_mean_diff) @ mean_diff)

        # 返回KID的平均值
    return kid_estimate / num_subsets