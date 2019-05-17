import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import seaborn as sns
from sklearn.datasets import make_moons

def plotClusters(label, y):
    plt.scatter(X[y==0,0], X[y==0,1], c='lightblue', marker='o', s=40, label='cluster 1')
    plt.scatter(X[y==1,0], X[y==1,1], c='red', marker='s', s=40, label='cluster 2')
    plt.xlabel(label)
    # plt.scatter

X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
sns.clustermap(X)

plt.figure()
plt.subplot(2,2,1)
X,y = make_moons(n_samples=300, noise=0.08, random_state=12)
plt.scatter(X[:,0],X[:,1])
plt.subplot(2,2,2)
plotClusters('kmeans',
             KMeans(n_clusters=2, max_iter=300, tol=1e-4, n_init=10).fit_predict(X))
plt.subplot(2,2,3)
plotClusters('agglomerative',
             AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit_predict(X))
plt.subplot(2,2,4)
plotClusters('DBSCAN',
             DBSCAN(eps=0.2, min_samples=5, metric='euclidean').fit_predict(X))


plt.show()