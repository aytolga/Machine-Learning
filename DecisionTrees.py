import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier   #Kütüphaneler Tanımlandı

my_data = pd.read_csv("drug200.csv", delimiter=",")
print("Our raw data:")
print(my_data[0:5])          #Veri setimizin ilk 5 satırını inceledik

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values #Eğitim Verilerimizi Aldık
print("Our training data:")
print(X[0:5])

"""
Bir sonraki aşamada elimizdeki verileri nümerik değerlere çevirme işlemi yapılmıştır
aksi taktirde algoritmamız doğru çalışmayacaktır.
"""

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])                #Kadın ve erkekleri numaralandırdık Kadınlar ---> 0 Erkekler -----> 1
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])  #Kalp atışı değerlerine 0 ile 2 arasında değerler verildi.
X[:,2] = le_BP.transform(X[:,2])       #high = 0, low = 1, normal = 2


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])   #Kolestrol değerleri numaralandırıldı. Bir yüksek bir normal değer bulunmaktadır.
X[:,3] = le_Chol.transform(X[:,3])
print("The Numeric Values of Training Data:")
print(X[0:5])

y = my_data["Drug"]
print("Our Test Data")
print(y[0:5])  #Test setimizi tanımladık bu aşamada bu ilaçların hangi hastalara uygun olduğunu tespit edeceğiz.

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3) #Eğitim ve test setlerimiz bölündü.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 5) #Decision Tree uygulandı.
"""
Fonksiyonun parametreleri, ilk parametre de entropi yöntemi tercih edildi max_depth = ne kadar dallanacağını gösteren bir değişken elimizdeki veriler yaş, cinsiyet,
kalp atış hızı ve kolestrol üzerinden ve bi tane daha farklı değişkenden gidileceği için 5 uygun olarak atandı. Bu aşamada kaç numaralı hastanın ne kullanması gerektiğini
tespit edeceğiz.
"""

drugTree.fit(X_trainset,y_trainset) #Test ve eğitim setini birleştirme.
predTree = drugTree.predict(X_testset) #X üzerinden tahmin.

print("The Result")
print (predTree [0:5])
print (y_testset [0:5])  #Sonuç

from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))   #Modelin doğruluk oranı.