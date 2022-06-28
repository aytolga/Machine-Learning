import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('nba2k-full.csv')
print(dataset)

x = dataset.iloc[:, 1].values.reshape(-1,1)



y = dataset.iloc[:, 8].str[1:].astype(np.int64).values.reshape(-1,1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

print(y.shape)
plt.scatter(x, y, color='blue')
plt.title('NBA Star Salary Datas')
plt.xlabel('Oyuncu Rankı')
plt.ylabel('Maaşı')
plt.show()


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('NBA Star Salary (Linear Regression)')
plt.xlabel('Oyuncu Rankı')
plt.ylabel('Maaşı')
plt.show()

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('NBA Star Salary (Polynomial Regression)')
plt.xlabel('Oyuncu Rankı')
plt.ylabel('Maaşı')
plt.show()