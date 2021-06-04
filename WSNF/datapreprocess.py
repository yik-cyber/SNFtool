import pyreadr
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# 用列均值替代缺失值
# 来源 https://blog.csdn.net/Weary_PJ/article/details/104064997
def fill_ndarray(mat):
    for i in range(mat.shape[1]):
        column = mat[:, i]
        # 计算列中空值个数
        nan_num = np.count_nonzero(column != column)
        if nan_num:
            column_not_nan = column[column == column]
            cur_mean = column_not_nan.mean()
            column[np.isnan(column)] = cur_mean
    return mat

# 检查是否有缺失值，没有返回true
def check_nan(mat):
    nan_num = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if(np.isnan(mat[i][j])):
                nan_num += 1
    return nan_num


saved_npy = False
saved_path = 'BRCA.mRNA.npy' 
def preprocess():
    
    if saved_npy:
        # 直接加载numpy数据
        data = np.load(saved_path, allow_pickle = True).astype('float')
    else:
        # 加载数据
        result = pyreadr.read_r('BRCA.mRNA.rda')
        mRNA = result['BRCA.mRNA']
        # 将数据作为一个矩阵，并转置，现在每一行是一个特征
        data = np.array(mRNA.values)
        # 只保留数据
        data = data[:, 1:]
        data = data.astype('float')
        np.save(saved_path, data)

    print('data shape: ', data.shape)
    # 用平均值填充缺失值
    data = fill_ndarray(data)

    if check_nan(data) == 0:
        print('no nan value\n')

    return data


if __name__ == '__main__':
    data = preprocess()
    np.save(saved_path, data.T)