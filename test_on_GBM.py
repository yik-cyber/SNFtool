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
print('*** silhouette score ***')
print("sklearn:", silhouette.silhouette_score(fusion, y_sklearn))
print("my spectral_cluster:", silhouette.silhouette_score(fusion, y_my_spectral_cluster))
print("my SpectralClustering:", silhouette.silhouette_score(fusion, y_my_SpectralClustering))

# 生存分析，计算3类survival曲线的p值
surv_df = pd.read_table(GBM_path + 'GLIO_Survival.txt', delim_whitespace=True)
from lifelines.statistics import multivariate_logrank_test
df = pd.DataFrame({
    'Survival': surv_df['Survival'],
    'Death': surv_df['Death'],
    'Label': y_my_SpectralClustering
})
results = multivariate_logrank_test(df['Survival'], df['Death'], df['Label'], weightings='peto')
print('\n*** p value ***')
print(results.p_value) # p < 0.05

# 绘制生存曲线
import matplotlib.pyplot as plt 
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

ax = plt.subplot(111)
cluster_num = 3
for i in range(cluster_num):
    idx = (df['Label'] == i)
    kmf.fit(df['Survival'][idx], df['Death'][idx], label='group' + str(i))
    kmf.plot_survival_function(ax=ax, ci_show=False)

plt.title('Lifespans for different groups')
plt.show()