"""

Author: Tolga Ay

Reach me from: aytolga@outlook.com

"""
import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

Categories = ['Kedi', 'Köpek'] #Örnek olması açısından öylesine yapılmış bir kategori işlemi
flat_data_arr = []  #Verilerin giriş dizisi
target_arr = []  #Verilerin çıkış dizisi
datadir = 'Verilerin_resimlerinin_bulunduğu_klasör_dizisi_genelden_başlayıp_içe_doğru_yazılmalı.'

#For döngüsü kullanılarak kategorilerin üstünde gezinme işlemi.
for i in Categories:

    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_resized = cv2.resize(img_array, (150, 150, 3))    #Kategorileri tanıtmak için yapılan eğitme işlemi
        flat_data_arr.append(img_resized.flatten())           #OpenCV kütüphanesindeki bazı işlemler resimleri eşitlemek ve okumak için kullanıldı.
        target_arr.append(Categories.index(i))

    print(f'loaded category:{i} successfully')


flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)  # Veri çerçevesi kodlamaları
df['Target'] = target
x = df.iloc[:, :-1]  # Giriş Verisi
y = df.iloc[:, -1]  # Çıkış Verisi

#SVM algoritması kullanılarak yapılan sınıflandırma işlemi
from sklearn import svm
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)


#Sınıflandırma ve SVM algoritmasından sonra verilerin eğitim ve test verisi olarak ayrılması
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')

y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")    #Modelin yaptığı tahminler ve doğru verinin bastırılarak incelenmesi.
print(np.array(y_test))

url=input('Enter URL of Image :')
img=cv2.imread(url)
plt.imshow(img)
plt.show()
img_resize=cv2.resize(img,(150,150,3))          #Resmin URL'sinin alınması ve eğitilen veriler gibi resize edilmesi
l=[img_resize.flatten()]                        #Ardından resim tahminin ne olduğunun ortaya çıkması.
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(l)[0]])