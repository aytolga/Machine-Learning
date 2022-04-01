
import pandas as pd


data = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')

X = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values        #Veri setimizin x ve y olarak ayrılması

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #Veriyi eğitim ve test seti olarak bölme

from sklearn.tree import DecisionTreeClassifier
decisionTreeObject = DecisionTreeClassifier()    #Karar ağacı algoritması kullanıldı
decisionTreeObject.fit(X_train,y_train)

dt_test_sonuc = decisionTreeObject.score(X_test, y_test)
print("Karar Ağacı Doğruluk (test_seti): ",round(dt_test_sonuc,2)) #Karar ağacı sonucu

from sklearn.ensemble import RandomForestClassifier
randomForestObject = RandomForestClassifier(n_estimators=10)  #Rastgele ormanlar algoritması kullanıldı
randomForestObject.fit(X_train, y_train)

df_test_sonuc = randomForestObject.score(X_test, y_test)
print("Random Forest Doğruluk (test_seti): ",round(df_test_sonuc,2))   #Rastgele ormanlar sonucu.

from sklearn.ensemble import BaggingClassifier
baggingObject = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
baggingObject.fit(X_train, y_train)   #Bagging yöntemiyle topluluk öğrenmesi her iki yönteme uygulandı.

baggingObject_sonuc = baggingObject.score(X_test, y_test)
print("Bagging Doğruluk (test_seti): ", round(baggingObject_sonuc,2)) #İşlemin sonucu.


