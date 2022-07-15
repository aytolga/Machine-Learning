from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel('Sepa1 Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.show()

km = KMeans(n_clusters = 3, random_state=21)
km.fit(X)


new_labels = km.labels_
fig, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)
axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)
axes[0].set_title('Veri', fontsize=18)
axes[1].set_title('Tahmin', fontsize=18)

plt.show()
