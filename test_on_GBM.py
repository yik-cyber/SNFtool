# 在 GBM 数据上测试基于 SNF 的聚类效果
# 数据放在 ./dataset/GBM 路径下
import numpy as np
import pandas as pd 
from SNF import SNF
from spectral_clustering import spectralClustering, CalNMI

# 载入基因、甲基化、miRNA三项数据
GBM_path = './dataset/GBM/'
file_names = ['GLIO_Gene_Expression.txt', 'GLIO_Methy_Expression.txt', 'GLIO_Mirna_Expression.txt']
features = [pd.read_table(GBM_path + filename, delim_whitespace=True) for filename in file_names]
features = [np.array(feature).T for feature in features]

# 图融合
dists = [SNF.dist2(feature, feature) for feature in features]
affinities = [SNF.affinityMatrix(dist) for dist in dists]
fusion = SNF.SNF(affinities)

# 谱聚类
import sklearn.cluster
y_sklearn = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(fusion)

from spectral_clustering import spectralClustering
y_my_spectral_cluster = spectralClustering.spectral_cluster(fusion, 3)
y_my_SpectralClustering = spectralClustering.SpectualClustering(fusion, 3)


# 计算 silhouette score
from SNF import silhouette
print("sklearn:", silhouette.silhouette_score(fusion, y_sklearn))
print("my spectral_cluster:", silhouette.silhouette_score(fusion, y_my_spectral_cluster))
print("my SpectralClustering:", silhouette.silhouette_score(fusion, y_my_SpectralClustering))