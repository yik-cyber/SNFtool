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
y_pred_wsnf = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(wsnf_fusion)
y_pred_snf = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(snf_fusion)
y_pred_anf = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(anf_fusion)
y_pred_gen = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(gen_fusion)
y_pred_miRNA = sklearn.cluster.SpectralClustering(n_clusters=3, affinity='precomputed').fit_predict(miRNA_fusion)

time = np.load('WSNF/data/time.npy', allow_pickle = True).astype('float64')
status = np.load('WSNF/data/status.npy', allow_pickle = True).astype('float64')

from lifelines.statistics import multivariate_logrank_test
df_wsnf = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred_wsnf
})
df_snf = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred_snf
})
df_anf = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred_anf
})
df_gen = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred_gen
})
df_miRNA = pd.DataFrame({
    'Survival': time,
    'Death': status,
    'Label': y_pred_miRNA
})

from SNF.silhouette import silhouette_score
results = multivariate_logrank_test(df_wsnf['Survival'], df_wsnf['Death'], df_wsnf['Label'], weightings='peto')
print('\n*** WSNF p value ***')
print(results.p_value) # p < 0.05
print('Silhouette: {}'.format(silhouette_score(wsnf_fusion, y_pred_wsnf)))

results = multivariate_logrank_test(df_snf['Survival'], df_snf['Death'], df_snf['Label'], weightings='peto')
print('\n*** SNF p value ***')
print(results.p_value) # p < 0.05
print('Silhouette: {}'.format(silhouette_score(snf_fusion, y_pred_snf)))

results = multivariate_logrank_test(df_anf['Survival'], df_anf['Death'], df_anf['Label'], weightings='peto')
print('\n*** ANF p value ***')
print(results.p_value) # p < 0.05
print('Silhouette: {}'.format(silhouette_score(anf_fusion, y_pred_anf)))

results = multivariate_logrank_test(df_gen['Survival'], df_gen['Death'], df_gen['Label'], weightings='peto')
print('\n*** gen p value ***')
print(results.p_value) # p < 0.05
print('Silhouette: {}'.format(silhouette_score(gen_fusion, y_pred_gen)))

results = multivariate_logrank_test(df_miRNA['Survival'], df_miRNA['Death'], df_miRNA['Label'], weightings='peto')
print('\n*** miRNA p value ***')
print(results.p_value) # p < 0.05
print('Silhouette: {}'.format(silhouette_score(miRNA_fusion, y_pred_miRNA)))