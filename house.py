import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df = df.drop('society', axis=1)
df['location'] = df['location'].factorize()[0]
df['size'] = df['size'].factorize()[0]
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

#imp = Imputer(missing_values='', strategy='mean', axis=0)
imp1 = Imputer(missing_values='NaN', strategy='mean', axis=0)
'''df['total_sqft'] = df['total_sqft'].values.reshape(-1,1)
df['total_sqft'] = imp1.fit_transform(df['total_sqft'])
df['total_sqft'] = df['total_sqft'].values.reshape(-1)'''

#print(df.head())
#print(df.columns)
#print(df.info())
#print(df['society'].value_counts())

X = df.drop('price', axis=1).values
Y = df['price'].values

#print(Y)

le_X_0 = LabelEncoder()
le_X_1 = LabelEncoder()
le_X_2 = LabelEncoder()
le_X_3 = LabelEncoder()

X[:, 0] = le_X_0.fit_transform(X[:, 0])
X[:, 1] = le_X_1.fit_transform(X[:, 1])
X[:, 2] = le_X_2.fit_transform(X[:, 2])
X[:, 3] = le_X_3.fit_transform(X[:, 3])

#X = imp.fit_transform(X)
X = imp1.fit_transform(X)

print(X[:,2])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 21)

regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
print(Y_pred)
'''print(accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))'''

def rmse(Y_pred, Y_test):
	error = np.square(np.log10(Y_pred+1) - np.log10(Y_test+1)).mean() ** 0.5
	acc = 1 - error
	return acc

print(rmse(Y_pred, Y_test))


#############################################

df2 = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
df2 = df2.drop(['price','society'], axis=1)
print(df2.head())

df2['size'] = df2['size'].factorize()[0]
df2['total_sqft'] = pd.to_numeric(df2['total_sqft'], errors='coerce')
print(df2.info())
df2 = df2.values

le_df2_0 = LabelEncoder()
le_df2_1 = LabelEncoder()
le_df2_2 = LabelEncoder()
le_df2_3 = LabelEncoder()

df2[:, 0] = le_df2_0.fit_transform(df2[:, 0])
df2[:, 1] = le_df2_1.fit_transform(df2[:, 1])
df2[:, 2] = le_df2_2.fit_transform(df2[:, 2])
df2[:, 3] = le_df2_3.fit_transform(df2[:, 3])


df2 = imp1.fit_transform(df2)

predictions = regressor.predict(df2)
print(predictions)

df3 = pd.DataFrame()
df3['price'] = predictions
print(df3.head())

df3.to_excel('Submission.xlsx', index= False)