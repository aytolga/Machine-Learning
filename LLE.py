from sklearn.datasets import load_iris
from sklearn.manifold import LocallyLinearEmbedding
data = load_iris()
X = data.data
print("LLE olmadan önceki boyutlar:")
print(X.shape)

embedding = LocallyLinearEmbedding(n_components=2)
X_transformed = embedding.fit_transform(X[:100])
print("LLE algoritmasından sonraki boyutlar:")
print(X_transformed.shape)
