import pandas as pd
import numpy as np

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

data = pd.read_csv("players_stats.csv")

data = data[['Pos','FG%','FT%','3P%','REB','AST','BLK','Height','Weight']]


data = data.replace(to_replace='C',value = '5')
data = data.replace(to_replace='PF',value = '4')
data = data.replace(to_replace='SF',value = '3')
data = data.replace(to_replace='SG',value = '2')
data = data.replace(to_replace='PG',value = '1')

clean_dataset(data)

print(data)

target = data['Pos']

del data['Pos']


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

from sklearn import svm
from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_train, y_train, cv=5)
y_pred = clf.predict(X_test)

print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print("Accuracy mean {}".format(scores.mean()*100))
print(scores)

JH = np.array([[50.2,90.5,40.7,200,350,32,196,95]])
LBJ = np.array([[48.8,71,35.4,340,511,49,202,113]])
GA = np.array([[49.1,74.1,15.6,542,207,55,207.5,100]])

deneme1 = clf.predict(JH)
deneme = clf.predict(LBJ)
deneme2 = clf.predict(GA)

print("Oyuncu: LeBron James Pozisyonu: {}".format(deneme))
print("Oyuncu: James Harden Pozisyonu: {}".format(deneme1))
print("Oyuncu: Giannis Antetekoumpo Pozisyonu: {}".format(deneme2))

ad = input("Oyuncu Adını Giriniz:")

fg = input("Oyuncunun FG yüzdesini giriniz:")
ft = input("Oyuncunun FT yüzdesini giriniz:")
tp = input("Oyuncunun 3P yüzdesini giriniz:")
trb = input("Oyuncunun sezon boyu olan ribaund sayısını giriniz:")
ast = input("Oyuncunun sezon boyu olan asist sayısını giriniz:")
blk = input("Oyuncunun sezon boyu olan blok sayısını giriniz:")
boy = input("Oyuncunun boyunu santimetre cinsinden giriniz:")
kilo = input("Oyuncunun kilosunu kilogram cinsinden giriniz:")

oyuncu = np.array([[fg, ft, tp, trb, ast, blk,boy,kilo]])

clf_oyuncu = clf.predict(oyuncu)

print("Oyuncu: {}".format(ad))
print("Pozisyonu: {}".format(clf_oyuncu))






