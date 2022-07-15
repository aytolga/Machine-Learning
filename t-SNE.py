from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
print("Iris verileri boyut azaltmadan önce:")
print(x.shape)

plt.scatter(x[:,0], x[:,1])
plt.show()

tsne = TSNE(n_components=2)
z = tsne.fit_transform(x)
print("Iris verileri boyut azaldıktan sonra:")
print(z.shape)

plt.scatter(z[:,0], z[:,1])
plt.show()


