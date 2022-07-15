from sklearn.datasets._samples_generator import make_blobs
X,y_true = make_blobs(n_samples=400,centers=4,cluster_std=0.60,random_state=0)
X = X[:,::-1]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X)

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c = labels, s=40,cmap='viridis')
plt.show()

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:,0],X[:,1],c = labels, s=40,cmap='viridis')
plt.show()