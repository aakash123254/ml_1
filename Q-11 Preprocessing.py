import pandas as pd
dataset = pd.read_csv('Data1.csv')
X = dataset.iloc[:,:-1].values #Takes all rows of all columns except the last column
Y = dataset.iloc[:,-1].values # Takes all rows of the last column
X
Y

print(dataset.columns)
dataset

dataset.info()
dataset.head()
dataset.tail()

#Row and column count
print(dataset.shape)


#Count missing values
dataset.isnull().sum().sort_values(ascending=False)
print(dataset.isnull().sum())


#Removing insufficient column
dataset_new = dataset.drop(['Age',], axis = 1)
dataset_new

#To measure the central tendency of variables
dataset_new.describe()

#To change column name
dataset.rename(index=str, columns={'Country' : 'Countries','Age' : 'age', 'Salary' : 'Sal','Purchased' : 'Purchased'}, inplace = True)

dataset
#Count missing values
dataset.isnull().sum().sort_values(ascending=False)

#Print the missing value column
dataset[dataset.isnull().any(axis=1)].head()

#Remove missing value rows
ds_new = dataset.dropna()
ds_new
ds_new.shape
ds_new.isnull().sum()

#To check datatype
ds_new.dtypes

# To convert as integer
ds_new.loc[:, 'age'] = ds_new['age'].astype('int64')

print(ds_new.dtypes)
print(ds_new)
# Imputing Mean, Median and Most_frequent
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
