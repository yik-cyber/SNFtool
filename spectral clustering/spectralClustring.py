import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def spectral_cluster(X, n_clusters=3, sigma=1): 
    '''
    n_cluster : cluster into n_cluster subset
    sigma: a parameter of the affinity matrix
    
    '''
    def affinity_matrix(X, sigma=1):
        N = len(X)
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
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
    
    eig_val, egi_vec = np.linalg.eig(NL)
    k_min_idx = np.argpartition(eig_val, K)
    F = egi_vec[k_min_idx[:K]]
    
    # type three, need to normalize the F
    """
    if type == 3:
      temp = F ** 2
      row_sum = np.sqrt(temp.sum(axis = 1))
      F = (F / row_sum[:, np.newaxis]).T
    """
    
    def renormalization(newX):
        Y = newX.copy()
        for i in range(len(newX)):
            norm = 0
            for j in newX[i]:
                norm += (newX[i] ** 2)
            norm = norm ** 0.5
            Y[i] /= norm
        return Y
    
    F = renormalization(F)
    

    kmeans = KMeans(n_clusters=K).fit(F)
    return kmeans.labels_
    


if __name__ == "__main__":

    iris = datasets.load_iris()

    X_reduced = PCA(n_components=2).fit_transform(iris.data)

    y = spectral_cluster(X_reduced, n_clusters=3, sigma=1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1)
    plt.show()