import pandas as pd
import quandl
import math
import numpy as np
import sklearn
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import metrics
import pickle

style.use('ggplot')

df=quandl.get('WIKI/GOOGL')
print(df.head())
df=df[['Open','High','Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT_Change']=(df['Adj. Close']-df['Open'])/df['Open']*100.0

df=df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]

forecast_col='Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out=int(math.ceil(0.1*len(df)))
print(forecast_out)

df['label']=df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

x=np.array(df.drop(['label'],axis=1))
x=preprocessing.scale(x)
x=x[:-forecast_out]
x_lately=x[-forecast_out:]


df.dropna(inplace=True)
y=np.array(df['label'])

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
print(y_train.shape)

clf=LinearRegression(n_jobs=-1)
#clf=svm.SVR(kernel='poly')

clf.fit(x_train, y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in=open('linearregression.pickle', 'rb')
clf=pickle.load(pickle_in)

accuracy=clf.score(x_test, y_test)
forecast_set=clf.predict(x_lately)
print(forecast_set, accuracy,forecast_out)

df['forecast']=np.nan
last_date=df.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix= last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)]+[i]
print(df.head())

df['Adj. Close'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


print(accuracy)

print(len(x), len(y))     
print(df.tail())
print(df.head())
