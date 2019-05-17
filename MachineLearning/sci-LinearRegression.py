__author__ = 'Kostya'
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X = df[['LSTAT']].values
y = df['MEDV'].values
plt.scatter(X,y)

r = LinearRegression()
quad = PolynomialFeatures(degree=2)
X_q = quad.fit_transform(X)
r.fit(X_q,y)
X_arg = np.arange(X.min(), X.max(),1)[:,np.newaxis]
plt.plot(X_arg,r.predict(quad.fit_transform(X_arg)), color='green', lw=2)

trip = PolynomialFeatures(degree=3)
X_t = trip.fit_transform(X)
r.fit(X_t,y)
plt.plot(X_arg,r.predict(trip.fit_transform(X_arg)), color='red', lw=2)

sns.pairplot(df[['LSTAT','INDUS','NOX','RM','MEDV','TAX','RAD']])

plt.show()