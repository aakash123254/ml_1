# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plit
import pandas as pd

dataset = pd.read_csv('C://Users//administrator//Desktop//firstt.csv')
dataset
print(dataset.columns)
dataset


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 4].values
x
y


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.NAN, strategy='mean')
imputer = imputer.fit(x[:, 3:4])
x[:, 3:4] = imputer.transform(x[:, 3:4])
x


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.NAN, strategy='median')
imputer = imputer.fit(x[:, 3:4])
x[:, 3:4] = imputer.transform(x[:, 3:4])
x

