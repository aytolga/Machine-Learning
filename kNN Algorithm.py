
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

irisData = load_iris()  #Sık kullanılan İris çiçeği verisi.
# Veri ve hedef ayrımı
X = irisData.data
y = irisData.target
# Eğitim ve test seti ayrımı.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
# Modelin doğruluk oranı için kodlamalar.
print("Doğruluk:")
print(knn.score(X_test, y_test))
neighbors = np.arange(1, 10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Uygun K değeri için bir for döngüsü.
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Eğitim ve test verilerinin doğruluk değerleri ile birleştirilmesi.
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Grafik Oluşturma
plt.plot(neighbors, train_accuracy, label='Deneme veriseti doğruluğu')
plt.plot(neighbors, test_accuracy, label='Eğitim veriseti doğruluğu')

plt.legend()
plt.xlabel('K')
plt.ylabel('Doğruluk')
plt.show()

