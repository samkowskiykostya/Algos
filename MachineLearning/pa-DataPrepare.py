import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

df.classlabel = LabelEncoder().fit_transform(df.classlabel)
print((OneHotEncoder(categorical_features=[0]).fit_transform(df[['classlabel','price']]).toarray())) #Encodes classlabel as 1/0
print((pd.get_dummies(df[['price', 'color', 'size']]))) #Encodes all categorical mantioned fields as 1/0

#Split test data
print((train_test_split(df[['color', 'size', 'price']].as_matrix(), df.classlabel.as_matrix(), test_size=0.33)))

#Scaling
print(('MinMax', MinMaxScaler().fit_transform(np.arange(0,1,0.2))))
print(('Standard', StandardScaler().fit_transform(np.arange(0,1,0.2))))

#Use RandomForest to evaluate params importance
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
            'Alcalinity of ash', 'Magnesium', 'Total phenols',
            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
            'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

f = RandomForestClassifier(n_estimators=1000,n_jobs=-1)
f.fit(X_train,y_train)
ids=np.argsort(f.feature_importances_)[::-1]

plt.bar(list(range(X_train.shape[1])), f.feature_importances_[ids], color='blue', align='center')
plt.xticks(list(range(X_train.shape[1])), df_wine.columns[1:][ids], rotation=90)
plt.tight_layout()
plt.show()

