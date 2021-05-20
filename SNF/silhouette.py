# 基于similarity的silhouette计算，可能有bug

import numpy as np

def silhouette_samples(similarity, labels):
    '''
    返回一个一维 array，为每个 sample 的 silhouette score
    '''
    n_samples = similarity.shape[0]
    clusters = np.unique(labels)

    intra_cluster = np.zeros(n_samples)
    for i in range(n_samples):
        mask = labels == labels[i]
        mask[i] = False
        if np.count_nonzero(mask) > 0:
            intra_cluster[i] = np.mean(similarity[i, mask])

    inter_cluster = np.array([np.max([np.mean(similarity[i, (labels == j)]) 
                                    for j in clusters if j != labels[i]])
                                    for i in range(n_samples)])
    
    sil_samples = (intra_cluster - inter_cluster) / np.maximum(intra_cluster, inter_cluster)
    return sil_samples


def silhouette_score(similarity, labels):
    '''
    基于相似度矩阵计算聚类的 silhouette score
    '''
    return np.mean(silhouette_samples(similarity, labels))