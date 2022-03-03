"""
The Linear Regression Algorithm


Author: Tolga AY


aytolga@outlook.com
"""
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("KidemMaas.csv")    #Our dataset (contact me for this dataset)

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values          #Seperation for dataset.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0) #Training for model.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)                #Linear Regression processing for trained data.
y_pred = regressor.predict(X_test)

data.plot(x='Kidem', y='Maas', style='.',color='green',ms= '10')
plt.title('Kıdeme Göre Maaşlar')
plt.xlabel('Kıdem(Yıl)')                   #Graph Drawing.
plt.ylabel('Maaş(Yıllık)')
plt.plot(X_test,y_test)
plt.show()