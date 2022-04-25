import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')

X = data.iloc[:, [2,3]].values


from sklearn.decomposition import PCA

pca =PCA(n_components=0.5)
pca.fit(X)
x_pca = pca.transform(X)


x2=pca.inverse_transform(x_pca)
plt.scatter(X[:,0], X[:,1], alpha=0.1)
plt.scatter(x2[:,0], x2[:,1], alpha=0.9)
plt.show()