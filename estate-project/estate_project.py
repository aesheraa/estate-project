
#Emlak fiyatlar�n� yapay zeka kullanarak tahmin etme projesi

#  MULTIPLE LINEAR REGRESSION
# Multiple Linear Regression'da birden fazla ba��ms�z(independent) de�i�kene kar��l�k bir ba��ml�(dependent) d�ei�ken bulunur.
# Linear Regression veriler aras�nda var olan korelasyonu(ili�kiyi) kullanarak yeni gelecek verileri tahmin etme modelidir. 
#Burada makine ��renimi bize veriler aras�ndaki bu ili�kiyi belirlememize yard�mc� olur ve bu sayede yeni verileri tahmin edebiliriz. 


#Linear regression veriler aras�nda var olan korelasyonu kullanarak yeni gelecek verileri tahmin etme modelidir. 
#Linear regression hesaplamas�nda tek bir �ndependent veriable vard�r. Y = a + bX

#Ben bu projede multiple linear regression hesaplamas� kullanaca��m. 
#Multiple linear regressionda ikiden fazla �ndependent veriable al�narak dependent veriable hesaplan�r.

#Y= a+ b1X1 + b2X2 + b3X3 +...

#Bu projede Y de�eri konut fiyat�n� temsil ediyor.
#Denklemdeki de�i�kenler oda say�s�, bina ya�� ve  alan. 
#Bu projede elimde train i�in kulland���m sadece 8 adet veri var. Ve test i�in de 3 veri kullanaca��m.


import pandas as pd
import matplotlib.pyplot as plt


# sklearn library
from sklearn import linear_model

#Verileri csv dosyas� olarak okutuyorum. Dataframe olarak import ediyorum pandasta Yapay zeka modelimi. 
#��ine dataframe'i fit ederken ilk �nce �ndependent veriablelari giriyorum sonra elde edilen etmek istedi�im �ey olan dependent veriable yani fiyat de�i�keni giriyorum.

# veri setimizi import ediyoruz, ayra� olarak noktal� virg�l oldu�u i�in bunu belirtiyoruz:
df = pd.read_csv("multilinearregression.csv",sep = ";")


# Veri setimizi g�relim ve do�ru import etti�iniz kontrol edelim:
df

df[['alan', 'odasayisi', 'binayasi']]

df['fiyat']


#Linear regression modelini tan�ml�yorum ve Prediction yap�yorum, yani fiyat�n� tahmin etmesini istedi�im evin verilerini giriyorum ve inceliyorum.

reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Prediction yapal�m..

reg.predict([[230,4,10]])


reg.predict([[230,6,0]])


reg.predict([[355,3,20]])


reg.predict([[230,4,10], [230,6,0], [355,3,20]])



reg.coef_


# Multiple Linear regression form�l�m�ze d�nersek hat�rlayal�m:
# y= a + b1X1 + b2X2 + b3X3 + ... form�l�m�zd�

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


#Train ettikten sonra sunucumu kapatt���mda kaybolmas�n� ve train i�lemini tekrar yapmay� istemiyorum o y�zden 
#E�itilen makine ��renmesi modelini kaydetme ve y�kleme k�sm� kald� bunun i�in pickle k�t�phanesini kullan�yorum, 
#reg model dosyam� open ile a��yorum, binary modda kaydetmemi sa�l�yor.

