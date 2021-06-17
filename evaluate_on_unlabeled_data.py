from numpy.core.numeric import allclose
import pandas as pd
import numpy as np
from WSNF.WSNF import *
from ANF.fusion import ANF_TEST, SINGLE_VIEW_MAT
import matplotlib.pyplot as plt

gen_exp = np.load('WSNF\data\gen_exp.npy', allow_pickle = True).astype('float64')
gen_ranking = np.load('WSNF\data\gen_ranking.npy', allow_pickle = True).astype('float64')
miRNAExp = np.load('WSNF\data\miRNAExp.npy', allow_pickle = True).astype('float64')
miRNA_ranking = np.load('WSNF\data\miRNA_ranking.npy', allow_pickle = True).astype('float64')

ranking1 = [gen_ranking, miRNA_ranking]
gbm = [gen_exp, miRNAExp]

wsnf_fusion = WSNF(datasets=gbm, feature_rankings=ranking1,
                   beta = 0.8,
                   K = 20,alpha = 0.5, t = 20)
snf_fusion = SNF_TEST(datasets=gbm, K = 20, alpha = 0.5, t = 20)
anf_fusion = ANF_TEST(datasets=gbm, K = 20, alpha=0.5)
gen_fusion, miRNA_fusion = SINGLE_VIEW_MAT(datasets=gbm, K=20, alpha=0.5)

import sklearn.cluster

graphs = [wsnf_fusion, snf_fusion, anf_fusion, gen_fusion, miRNA_fusion]
names = ['WSNF', 'SNF', 'ANF', 'gen', 'miRNA']
time = np.load('WSNF/data/time.npy', allow_pickle = True).astype('float64')
status = np.load('WSNF/data/status.npy', allow_pickle = True).astype('float64')
from lifelines.statistics import multivariate_logrank_test

from SNF.silhouette import silhouette_score, silhouette_plot
from SNF.survival_analysis import plot_lifeline

y_pred_list = []
df_list = []
for name, graph in zip(names, graphs):
    y_pred = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(graph)
    y_pred_list.append(y_pred)
    df = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred
    })
    df_list.append(df)
    #print(df.info())


import SNF.snfpy_sil
fig, axs = plt.subplots(2,3)
i = 0
for name, graph, y_pred in zip(names, graphs, y_pred_list):
    print('\n*** {} ***'.format(name))
    print('Silhouette: {}'.format(silhouette_score(graph, y_pred)))
    print(SNF.snfpy_sil.silhouette_score(graph, y_pred))
    silhouette_plot(graph, y_pred, axs[i//3][i%3], title=name)
    i += 1

plt.show()


fig, axs = plt.subplots(2,3)
i = 0
for name, graph, y_pred, df in zip(names, graphs, y_pred_list, df_list):
    results = multivariate_logrank_test(df['Survival'], df['Label'], df['Death'], weightings='peto')
    print('\n*** {} ***'.format(name))
    print('p value:', results.p_value) # p < 0.05

    plot_lifeline(df, len(np.unique(y_pred)), axs[i//3][i%3], name + ' p value = {:.5f}'.format(results.p_value))
    i += 1

plt.show()
