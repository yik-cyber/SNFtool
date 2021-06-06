from SNF import *
import pandas as pd
import numpy as np


def MAD(mat):
    '''
    计算feature的absolute median值
    MAD(fi) = Median(Xi - Median(Xi))
    '''
    # print('mat shape: ', mat.shape)
    # MAD(fi)
    # median (feature, )
    median = np.median(mat, axis=1)
    median_matrix = median.copy().reshape(-1, 1)
    for i in range(mat.shape[1] - 1):
        median_matrix = np.append(median_matrix, median.reshape(-1, 1), axis=1)
    
    abs_median_matrix = np.abs(mat - median_matrix)
    return np.median(abs_median_matrix, axis=1)


def DistanceWeighted(X, weight):
    '''
    对于特征矩阵X，在计算distance中乘以weight
    X (patient, feature)
    稍微高效一点的实现
    '''
    assert (X.shape[1] == len(weight))
    diag_weight = np.diag(weight)
    XY = X @ diag_weight @ X.T

    # 取出对角元素
    XY_diag = np.diag(XY)
    # 把对角元素复制给每一列
    XX = np.empty(XY.shape)
    for i in range(XY.shape[0]):
        XX[i, :] = XY_diag
    res = np.sqrt(XX + XX.T - 2 * XY)
    return res


def EuclideanDistance(X):
    '''
    欧式距离
    X (patient, featurn)
    '''
    XY = X @ X.T
    XY_diag = np.diag(XY)
    XX = np.empty(XY.shape)
    for i in range(XY.shape[0]):
        XX[i, :] = XY_diag
    res = np.sqrt(XX + XX.T - 2 * XY)
    return res


def TrivalDistanceWeighted(X, weight):
    '''
    对于特征矩阵X，在计算distance中乘以weight
    X (patient, feature)
    一个trival的实现，用于检查
    '''
    assert (X.shape[1] == len(weight))
    patient_num = X.shape[0]
    feature_num = X.shape[1]
    res = np.empty((patient_num, patient_num))
    for i in range(patient_num):
        for j in range(patient_num):
            temp = 0
            for k in range(feature_num):
                temp += weight[k] * ((X[i][k] - X[j][k]) ** 2)
            res[i][j] = np.sqrt(temp)
    return res

def SNF_TEST(datasets, K, alpha, t):
    '''
    用于测试的SNF
    '''
    W_temp = []
    for data in datasets:
        distance = EuclideanDistance(data.T)
        afft_mat = affinityMatrix(distance, K, alpha)
        W_temp.append(afft_mat)

    W = SNF(W_temp, K=K, t=t)
    return W


def WSNF(datasets, feature_rankings, beta, K, alpha, t):
    '''
    weighted SNF
    与SNF不同之处在于计算距离时加入了weight
    '''    
    assert (len(datasets) == len(feature_rankings))
    W_temp = []
    for i in range(len(datasets)):
        # MADn(fi)
        data = datasets[i]
        mads = MAD(data)
        mads = mads / np.sum(mads)

        # Rn(fi)
        feature_ranking1 = feature_rankings[i]
        feature_ranking1 = feature_ranking1 / np.sum(feature_ranking1)

        # weight = beta * Rn(fi) + (1-beta)*MADn(fi)
        weight = beta * feature_ranking1 + (1 - beta) * mads
        # 计算patients距离
        distance = DistanceWeighted(data.T, weight)
        # 根据距离计算affini矩阵
        aff_mat = affinityMatrix(distance, K, alpha)
        W_temp.append(aff_mat)

    W = SNF(W_temp, K=K, t=t)
    return W