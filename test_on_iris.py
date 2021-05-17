from SNF import SNF
from spectral_clustering import spectralClustering
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans

print('*** concat & PCA2 + spectral ***')
iris = datasets.load_iris()
X_reduced = PCA(n_components=2).fit_transform(iris.data)
y = spectralClustering.spectral_cluster(X_reduced, n_clusters=3, sigma=1)
print('NMI:' , normalized_mutual_info_score(y, iris.target))

print('*** concat + kmeans ***')
y = KMeans(n_clusters=3).fit(iris.data).labels_
print('NMI:', normalized_mutual_info_score(y, iris.target))

print('*** SNF + spectral ***')
dists = [SNF.dist2(iris.data[:,i], iris.data[:,i]) for i in range(4)]
affinities = [SNF.affinityMatrix(dist) for dist in dists]
fusioned = SNF.SNF(affinities)
y = spectralClustering.SpectualClustering(fusioned, K=3)
print('NMI:', normalized_mutual_info_score(y, iris.target))