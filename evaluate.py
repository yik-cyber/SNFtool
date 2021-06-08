from numpy.core.numeric import allclose
import pandas as pd
import numpy as np
from WSNF.WSNF import *
from ANF.fusion import ANF_TEST, SINGLE_VIEW_MAT

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
names = ['wsnf', 'snf', 'anf', 'gen', 'miRNA']
time = np.load('WSNF/data/time.npy', allow_pickle = True).astype('float64')
status = np.load('WSNF/data/status.npy', allow_pickle = True).astype('float64')
from lifelines.statistics import multivariate_logrank_test

from SNF.silhouette import silhouette_score

for name, graph in zip(names, graphs):
    y_pred = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(graph)
    df = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred
    })
    results = multivariate_logrank_test(df['Survival'], df['Death'], df['Label'], weightings='peto')
    print('\n*** {} ***'.format(name))
    print('p value:', results.p_value) # p < 0.05
    print('Silhouette: {}'.format(silhouette_score(graph, y_pred)))