from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def My_Kmeans(X, K):
  sample_num = X.shape[0]
  index = np.arange(sample_num)
  np.random.shuffle(index)
  index = index[:K]
  # 从X中随机选取k个点作为中心点
  central_points = X[index]
  y = np.arange(sample_num)

  while True:
    new_y = np.arange(sample_num)
    changed = False
    for i, xi in enumerate(X):
      # xi对应最小距离中心点的下标
      new_y[i] = np.argmin([np.linalg.norm(xi - cj) for cj in central_points])
      if new_y[i] != y[i]:
          changed = True
    if not changed:
      break
    for j in range(K):
      # 平均值更新中心点
      central_points[j] = np.mean(X[np.where(new_y == j)], axis=0)
    y = new_y.copy()

  return y 



def spectral_cluster(X, n_clusters=3, sigma=1): 
    def affinity_matrix(X, sigma=1):
        n = len(X)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                A[i][j] = np.exp(-np.linalg.norm(X[i] - X[j])**2 / (2 * sigma ** 2))
                A[j][i] = A[i][j]
        return A
    
    A = affinity_matrix(X, sigma)
    
    return SpectualClustering(A, n_clusters)


def SpectualClustering(affinity, K, type = 3):
    ###This function implements the famous spectral clustering algorithms. There are three variants. The default one is the third type. 
    ###THe inputs are as follows:
  
    #affinity: the similarity matrix;
    #K: the number of clusters
    #type: indicators of variants of spectral clustering
    
    affinity = np.float64(affinity)
    d = np.sum(affinity, axis=1)
    #将求和为0的项设置为eps，同时所有项的type都为float64
    d[d == 0] = np.finfo(np.float64).eps
    D = np.diag(d)

    L = D - affinity
    ## Ratiocut argmin(tr(HtLH)) HtH = I
    if type == 1:
      NL = L 
    ## Normalize L with sqrt(di)
    elif type == 2:
      Di = np.diag(1/d)
      NL = Di @ L
    ## Normalize L with sqrt(di*dj)
    ## Ncut argmin(tr(FtD^(-1/2)LD^(-1/2)F) FtF = I
    elif type == 3:
      Di = np.diag(1.0 / np.sqrt(d))
      NL = Di @ L @ Di
    
    eig_val, eig_vec = np.linalg.eig(NL)
    index = np.argsort(eig_val)[:K]
    F = eig_vec[:, index]
    
    # type three, need to normalize the F
    if type == 3:
      temp = F ** 2
      row_sum = np.sqrt(temp.sum(axis = 1))
      F = (F / row_sum[:, np.newaxis])
    

    # kmeans = KMeans(n_clusters=K).fit(F)
    # return kmeans.labels_
    return My_Kmeans(F, K)
    


if __name__ == "__main__":
    iris = datasets.load_iris()
    X_reduced = PCA(n_components=2).fit_transform(iris.data)
    y = spectral_cluster(X_reduced, n_clusters=3, sigma=1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
    plt.show()