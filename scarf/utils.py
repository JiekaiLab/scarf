import torch
import numpy as np
import scipy
import pickle
import argparse
import os
import pandas as pd
from geomloss import SamplesLoss
import torch.nn as nn
import torch.nn.functional as F
import json

def load_model_with_index(model_name_or_path):
    """
    使用索引文件加载 PyTorch 模型。

    Args:
        model_name_or_path (str): 模型的路径或名称。

    Returns:
        torch.nn.Module: 加载的模型。
    """
    # 加载索引文件
    index_file_path = os.path.join(model_name_or_path, 'pytorch_model.bin.index.json')
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)

    # 获取权重文件列表
    weight_files = index_data['weight_map'].values()
    weight_files = list(set(weight_files))  # 去重

    # 加载所有权重文件
    state_dict = {}
    for weight_file in weight_files:
        weight_path = os.path.join(model_name_or_path, weight_file)
        checkpoint = torch.load(weight_path)
        state_dict.update(checkpoint)
    return state_dict

class OptimalTransportLoss(nn.Module):
    def __init__(self):
        super(OptimalTransportLoss, self).__init__()
        # 使用SamplesLoss来计算Wasserstein距离
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.8)

    def forward(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)

        # x and y are the features from two modalities, shape (b, d)
        # 计算Wasserstein距离
        return self.loss(x, y)


class Focal_Loss(torch.nn.Module):
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.5, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        preds = torch.sigmoid(preds.float())
        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds.float() + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds.float() + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)


def cosine_similarity_matrix(matrix1, matrix2, dim=1):
    """
    计算两个二维矩阵的余弦相似度矩阵

    参数：
    - matrix1: 第一个二维矩阵，形状为 (n, d)
    - matrix2: 第二个二维矩阵，形状为 (m, d)
    - dim: 沿哪个维度计算相似度，默认为 1（按行计算）

    返回：
    - similarity_matrix: 余弦相似度矩阵，形状为 (n, m)
    """
    matrix1 = torch.from_numpy(matrix1)
    matrix2 = torch.from_numpy(matrix2)
    # 归一化两个矩阵
    matrix1_normalized = F.normalize(matrix1, p=2, dim=dim)
    matrix2_normalized = F.normalize(matrix2, p=2, dim=dim)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(matrix1_normalized, matrix2_normalized.transpose(0, 1))

    return similarity_matrix


def sinkhorn_knopp(a, b, C, epsilon, max_iter=100):
    """
    使用 Sinkhorn-Knopp 算法计算最优传输矩阵

    参数：
    - a: 源分布的权重向量，形状为 (n,)
    - b: 目标分布的权重向量，形状为 (m,)
    - C: 成本矩阵，形状为 (n, m)
    - epsilon: 正则化参数
    - max_iter: 最大迭代次数

    返回：
    - P: 最优传输矩阵，形状为 (n, m)
    """
    # 初始化传输矩阵
    P = torch.exp(-C / epsilon)

    # 归一化
    P = P / P.sum()

    # Sinkhorn-Knopp 算法
    for _ in range(max_iter):
        # 更新行归一化因子
        u = a / P.sum(dim=1)
        P = P * u.unsqueeze(1)

        # 更新列归一化因子
        v = b / P.sum(dim=0)
        P = P * v.unsqueeze(0)

    return P

# Calculate metrics
def calculate_foscttm(rna_cell_embedding, atac_cell_embedding):
    d = scipy.spatial.distance_matrix(rna_cell_embedding, atac_cell_embedding)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    foscttm = np.mean(foscttm_x + foscttm_y) / 2

    num_samples = len(rna_cell_embedding)
    a = torch.ones(num_samples) / num_samples
    b = torch.ones(num_samples) / num_samples
    similarity_matrix = cosine_similarity_matrix(rna_cell_embedding, atac_cell_embedding)
    # 将相似度矩阵转换为成本矩阵，这里使用1 - similarity_matrix作为成本
    C = 1 - similarity_matrix
    # 设置正则化参数
    epsilon = 0.01
    # 计算最优传输矩阵
    pi = sinkhorn_knopp(a, b, C, epsilon).numpy()
    foscttm_x_ot = (pi > np.expand_dims(np.diag(pi), axis=1)).mean(axis=1)
    foscttm_y_ot = (pi > np.expand_dims(np.diag(pi), axis=0)).mean(axis=0)
    foscttm_ot = np.mean(foscttm_x_ot + foscttm_y_ot) / 2
    return foscttm, np.mean(foscttm_x), np.mean(foscttm_y), \
           foscttm_ot, np.mean(foscttm_x_ot), np.mean(foscttm_y_ot)

@torch.no_grad()
def calculate_top_matching(rna_cell_embedding, atac_cell_embedding, ntop=100, logit_scale=None):
    d = scipy.spatial.distance_matrix(rna_cell_embedding, atac_cell_embedding)
    top_matching_x = (d < np.expand_dims(np.diag(d), axis=1)).sum(axis=1) <= (ntop - 1)
    top_matching_y = (d < np.expand_dims(np.diag(d), axis=0)).sum(axis=0) <= (ntop - 1)
    top_matching = (top_matching_x.mean() + top_matching_y.mean()) / 2
    # calculat format2
    # similarity = torch.from_numpy(rna_cell_embedding).to('cpu') @ torch.from_numpy(atac_cell_embedding).t().to('cpu')
    # mean_sim = ((similarity.softmax(0) + similarity.softmax(1)) / 2).detach().numpy()
    # top_matching_x_new = (mean_sim < np.expand_dims(np.diag(mean_sim), axis=1)).sum(axis=1) <= (ntop - 1)
    # top_matching_y_new = (mean_sim < np.expand_dims(np.diag(mean_sim), axis=0)).sum(axis=0) <= (ntop - 1)
    # top_matching_new = (top_matching_x_new.mean() + top_matching_y_new.mean()) / 2
    num_samples = len(rna_cell_embedding)
    a = torch.ones(num_samples) / num_samples
    b = torch.ones(num_samples) / num_samples
    similarity_matrix = cosine_similarity_matrix(rna_cell_embedding, atac_cell_embedding)
    # 将相似度矩阵转换为成本矩阵，这里使用1 - similarity_matrix作为成本
    C = 1 - similarity_matrix
    # 设置正则化参数
    epsilon = 0.01
    # 计算最优传输矩阵
    pi = sinkhorn_knopp(a, b, C, epsilon).numpy()
    top_matching_x_new = (pi > np.expand_dims(np.diag(pi), axis=1)).sum(axis=1) <= (ntop - 1)
    top_matching_y_new = (pi > np.expand_dims(np.diag(pi), axis=0)).sum(axis=0) <= (ntop - 1)
    top_matching_new = (top_matching_x_new.mean() + top_matching_y_new.mean()) / 2

    return top_matching.mean(), np.mean(top_matching_x), np.mean(top_matching_y),\
           top_matching_new.mean(), np.mean(top_matching_x_new), np.mean(top_matching_y_new),


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_object(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    object_gather = [None
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(object_gather, tensor)
    output = []
    for i in object_gather:
        output.extend(i)
    return output


def str2bool(v):
    v = v.rstrip(',')
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(v.lower())
        raise argparse.ArgumentTypeError('Boolean value expected.')