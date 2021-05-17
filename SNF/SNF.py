import numpy as np
import copy
from scipy.stats import norm

def dist2(X, C):
    '''
    求 X 的每个行向量和 C 的每个行向量欧氏距离的平方
    '''
    nrows = X.shape[0]
    ncols = C.shape[0]
    res = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            res[i, j] = np.inner(X[i]-C[j], X[i]-C[j])
    res[res < 0] = 0
    return res


def affinityMatrix(Diff, K=20, sigma=0.5):
    '''
    根据距离矩阵 Diff 建立相似网络
    '''
    N = Diff.shape[0]

    Diff = (Diff + Diff.T) / 2
    np.fill_diagonal(Diff, 0)

    eps = np.finfo(np.float64).eps

    means = np.array([np.mean(sorted(row)[1:K+1]) for row in Diff]) + eps
    #np.savetxt('means_py.csv', means, fmt = '%f', delimiter=',')
    Sig = np.add.outer(means, means) / 3 + Diff / 3 + eps
    Sig[Sig <= eps] = eps
    
    densities = norm.pdf(Diff, 0, sigma * Sig)

    W = (densities + densities.T) / 2
    return W

def normalize(X):
    '''
    每行除以行和
    '''
    return X / np.sum(X, axis=1).reshape(-1, 1)

def dominateset(xx, KK=20):
    '''
    保留边最大的 KK 个邻居 
    '''
    A = np.copy(xx)
    idx = np.argsort(A, axis = 1)
    for i in range(A.shape[0]):
        A[i, idx[i, :A.shape[1]-KK]] = 0
    return normalize(A)

def SNF(Wall, K=20, t=20):
    '''
    Wall 是相似网络的 list，
    返回融合网络
    '''
    Wall = copy.deepcopy(Wall)
    LW = len(Wall)

    newW = [0 for i in range(LW)]
    nextW = [0 for i in range(LW)]

    # normalize
    for i in range(LW):
        Wall[i] = normalize(Wall[i])
        Wall[i] = (Wall[i] + Wall[i].T)/2

    # Local transition matrix
    for i in range(LW):
        newW[i] = dominateset(Wall[i], K)
        #np.savetxt('dom_py_' + str(i) + '.csv', newW[i], fmt = '%f', delimiter=',')

    # perfrom the diffusion for t times
    for i in range(t):
        for j in range(LW):
            WJ = Wall[j]
            sumWJ = np.zeros(WJ.shape)
            for k in range(LW):
                if k != j:
                    sumWJ = sumWJ + Wall[k]
            nextW[j] = newW[j] @ (sumWJ / (LW-1)) @ newW[j].T

        # normalize new obtained networks
        for j in range(LW):
            Wall[j] = nextW[j] + np.eye(Wall[j].shape[0])
            Wall[j] = (Wall[j] + Wall[j].T) / 2

    # construct the combined affinity matrix by summing diffused matrices
    W = np.zeros(Wall[0].shape)
    for i in range(LW):
        W = W + Wall[i]
    W = W / LW
    W = normalize(W)
    W = (W + W.T + np.eye(W.shape[0])) / 2

    return W