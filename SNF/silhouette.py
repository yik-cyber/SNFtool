# 基于similarity的silhouette计算
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def silhouette_plot(similarity, cluster_labels, ax1, title = ''):
    sample_silhouette_values = silhouette_samples(similarity, cluster_labels)
    n_clusters = len(np.unique(cluster_labels))
    silhouette_avg = silhouette_score(similarity, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(title)
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])