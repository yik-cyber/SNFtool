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
from sklearn.cluster import SpectralClustering
y_sklearn = SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(fusion)

# 计算 silhouette score
from sklearn.metrics import silhouette_score
cat_feature = np.concatenate(features, axis = 1) # 样本间距离设为三个特征连接起来的欧氏距离，原文的设定可能不同
score = silhouette_score(cat_feature, y_sklearn)
print("silhouette score:", score) # 0.093