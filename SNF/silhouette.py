# 基于similarity的silhouette计算，可能有bug

import numpy as np 

def silhouette_samples(similarity, labels):
    '''
    返回一个一维 array，为每个 sample 的 silhouette score
    '''
    n_samples = similarity.shape[0]
    clusters = np.unique(labels)
    n_clusters = len(clusters)
    intra_cluster = np.array([np.mean(similarity[i, labels == labels[i]]) for i in range(n_samples)])
    inter_cluster = np.array([np.max([np.mean(similarity[i, labels == j]) 
                                    for j in clusters if j != labels[i]])
                                    for i in range(n_samples)])

    sil_samples = (intra_cluster - inter_cluster) / np.max((intra_cluster, inter_cluster), axis = 0)
    return sil_samples


def silhouette_score(similarity, labels):
    '''
    基于相似度矩阵计算聚类的 silhouette score
    '''
    return np.mean(silhouette_samples(similarity, labels))