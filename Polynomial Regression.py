"""
The Polynomial Regression Algorithm


Author: Tolga AY


aytolga@outlook.com
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("PozisyonSeviyeMaas.csv")    #Our dataset (contact me for this dataset)
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values         #Seperation of dataset with iloc function.
print(X)
print(y)

from sklearn.linear_model import LinearRegression      #Implenation of libraries for machine learning.
from sklearn.preprocessing import PolynomialFeatures

lr = LinearRegression()           #Calling LinearRegression class for training
polinom = PolynomialFeatures(3)   #Right degree for polynomial for prediction curve.

maaslarpol = polinom.fit_transform(X)    #Transforming to a polynomial
lr.fit(maaslarpol, y)
predict = lr.predict(maaslarpol)        #Training for model.

plt.title('Kıdeme Göre Maaşlar')
plt.xlabel('Pozisyon')                   #For graph.
plt.ylabel('Maaş(Yıllık)')
plt.scatter(X, y, color = "red")
plt.plot(X,predict,color="blue")
plt.show()
