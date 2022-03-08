import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Rastgele 25 tane mavi ve kırmızı noktalar üretelim.
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# Kırmızıları 1, mavileri 2 ile işaretleyelim.
responses = np.random.randint(0,2,(25,1)).astype(np.float32)
# Kırmızı komşuları birleştirelim.
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],80,'r','^')
# Mavi komşuları birleştirelim
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],80,'b','s')

# Yeni veri oluşturalım ve grafikte rastgele bir yere ekleyelim
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

# OpenCV sayesinde kNN uygulayalım ve sonuçları inceleyelim.
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)
print( "result:  {}\n".format(results) )
print( "neighbours:  {}\n".format(neighbours) )
print( "distance:  {}\n".format(dist) )
plt.show()