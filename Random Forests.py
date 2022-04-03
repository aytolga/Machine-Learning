import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("PozisyonSeviyeMaas.csv")  #Veri setimizin alınışı.
print(dataset.head())

#Değişkenler oluşturuldu burada X değişkeni seviyeyi Y değişkeni ise alınan maaşı temsil etmektedir.
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Random Forests algoritmasındaki regresyon modeli kullanılarak veri setimiz eğitildi.
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0) #n_estimators parametresi algoritmada kaç tane orman olacağını belirtmektedir bu veri seti için 10 ağaç en uygunudur.
regressor.fit(X, y)

#Modeli incelemek için bir grafik çizelim.
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'black')
plt.title('Random Forests Algoritması')
plt.xlabel('Ünvan')
plt.ylabel('Maaş')
plt.show()
