from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('datasets/titanic.csv', sep='\t')
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

lr = LogisticRegression()
lr.fit(df[['Sex','Pclass']], df['Survived'])
print(lr.intercept_, lr.coef_)