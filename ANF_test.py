# 比较单个特征、SNF、ANF聚类效果
# 特征为 fpkm（基因表达）、methy450（甲基化）、miRNA

import numpy as np 
import pandas as pd 
cancer_types = ['adrenal_gland', 'colorectal', 'kidney', 'lung', 'uterus']
feature_types = ['fpkm', 'methy450', 'mirnas']
datapath = './dataset/ANFdata/'
id2subtype = pd.read_table(datapath + 'project_ids.txt', sep=' ')
id2subtype.x = pd.factorize(id2subtype.x)[0]
id2subtype = id2subtype['x']

from ANF import ANF, clustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from SNF import SNF

ANF_NMI = []
SNF_NMI = []
single_NMI = [[],[],[]]

for cancer_type in cancer_types:
    affinities = [pd.read_table(datapath + cancer_type + '_' + featuretype + '_.txt', sep=' ') 
        for featuretype in feature_types]
    graphs = [np.array(affinity) for affinity in affinities]

    y_true = [id2subtype[id] for id in affinities[0].index]
    n_cluster = len(np.unique(y_true))

    # ANF
    fusion = ANF.affinity_graph_fusion(graphs)
    fusion = (fusion + fusion.T)/2
    y_pred = SpectralClustering(n_clusters=n_cluster, random_state=0, affinity='precomputed').fit_predict(fusion)
    NMI = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ANF_NMI.append(NMI)

    # SNF
    fusion = SNF.SNF(graphs)
    y_pred = SpectralClustering(n_clusters=n_cluster, random_state=0, affinity='precomputed').fit_predict(fusion)
    NMI = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    SNF_NMI.append(NMI)

    # single feature
    for i in range(len(affinities)):
        y_pred = SpectralClustering(n_clusters=n_cluster, random_state=0, affinity='precomputed').fit_predict(affinities[i])
        NMI = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
        single_NMI[i].append(NMI)

import matplotlib.pyplot as plt

NMI_df = pd.DataFrame({
    'fpkm': single_NMI[0],
    'methy450': single_NMI[1],
    'mirnas': single_NMI[2],
    'SNF': SNF_NMI,
    'ANF': ANF_NMI,
    'cancer type': cancer_types
})

NMI_df = NMI_df.set_index('cancer type')
NMI_df.plot(kind='bar', figsize=(12,6), rot=0)
plt.show()