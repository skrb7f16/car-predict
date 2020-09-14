import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('train.csv')
df.drop('v.id',inplace=True,axis=1)
df.drop('on road old',inplace=True,axis=1)
df.drop('on road now',inplace=True,axis=1)
df.drop('rating',inplace=True,axis=1)
df.drop('condition',inplace=True,axis=1)
df.drop('torque',inplace=True,axis=1)


X=df[['years', 'km', 'economy', 'top speed', 'hp']]

y=df['current price']


from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(X,y)

pickle.dump(lm,open('model.pkl','wb'))
