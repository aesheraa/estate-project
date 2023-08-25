
#Emlak fiyatlarýný yapay zeka kullanarak tahmin etme projesi

#  MULTIPLE LINEAR REGRESSION
# Multiple Linear Regression'da birden fazla baðýmsýz(independent) deðiþkene karþýlýk bir baðýmlý(dependent) dðeiþken bulunur.
# Linear Regression veriler arasýnda var olan korelasyonu(iliþkiyi) kullanarak yeni gelecek verileri tahmin etme modelidir. 
#Burada makine öðrenimi bize veriler arasýndaki bu iliþkiyi belirlememize yardýmcý olur ve bu sayede yeni verileri tahmin edebiliriz. 


#Linear regression veriler arasýnda var olan korelasyonu kullanarak yeni gelecek verileri tahmin etme modelidir. 
#Linear regression hesaplamasýnda tek bir Ýndependent veriable vardýr. Y = a + bX

#Ben bu projede multiple linear regression hesaplamasý kullanacaðým. 
#Multiple linear regressionda ikiden fazla Ýndependent veriable alýnarak dependent veriable hesaplanýr.

#Y= a+ b1X1 + b2X2 + b3X3 +...

#Bu projede Y deðeri konut fiyatýný temsil ediyor.
#Denklemdeki deðiþkenler oda sayýsý, bina yaþý ve  alan. 
#Bu projede elimde train için kullandýðým sadece 8 adet veri var. Ve test için de 3 veri kullanacaðým.


import pandas as pd
import matplotlib.pyplot as plt


# sklearn library
from sklearn import linear_model

#Verileri csv dosyasý olarak okutuyorum. Dataframe olarak import ediyorum pandasta Yapay zeka modelimi. 
#Ýçine dataframe'i fit ederken ilk önce Ýndependent veriablelari giriyorum sonra elde edilen etmek istediðim þey olan dependent veriable yani fiyat deðiþkeni giriyorum.

# veri setimizi import ediyoruz, ayraç olarak noktalý virgül olduðu için bunu belirtiyoruz:
df = pd.read_csv("multilinearregression.csv",sep = ";")


# Veri setimizi görelim ve doðru import ettiðiniz kontrol edelim:
df

df[['alan', 'odasayisi', 'binayasi']]

df['fiyat']


#Linear regression modelini tanýmlýyorum ve Prediction yapýyorum, yani fiyatýný tahmin etmesini istediðim evin verilerini giriyorum ve inceliyorum.

reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Prediction yapalým..

reg.predict([[230,4,10]])


reg.predict([[230,6,0]])


reg.predict([[355,3,20]])


reg.predict([[230,4,10], [230,6,0], [355,3,20]])



reg.coef_


# Multiple Linear regression formülümüze dönersek hatýrlayalým:
# y= a + b1X1 + b2X2 + b3X3 + ... formülümüzdü

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 230
x2 = 4
x3 = 10
y = a + b1*x1 + b2*x2 + b3*x3

y


#Train ettikten sonra sunucumu kapattýðýmda kaybolmasýný ve train iþlemini tekrar yapmayý istemiyorum o yüzden 
#Eðitilen makine öðrenmesi modelini kaydetme ve yükleme kýsmý kaldý bunun için pickle kütüphanesini kullanýyorum, 
#reg model dosyamý open ile açýyorum, binary modda kaydetmemi saðlýyor.

