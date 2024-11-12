# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:33:09 2024

@author: Parthiv
"""

import numpy as np
import matplotlib.pyplot as plit
import pandas as pd

dataset = pd.read_csv('C:/Users/Parthiv/Desktop/Other/MCA/sem3/MachineLearning/prediction.csv')
dataset
print(dataset.columns)
dataset

# x = dataset.iloc[:,:-1].values
x = dataset.iloc[:,:-1].values
x
y = dataset.iloc[:,4].values
y


#  x[:, 3:4]  => : for all the raw (loop) && 3:4 for the perticuller column   x[raw,columan]

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
# imputer = imputer.fit(x[:, 3:4], x[:,4:5])
imputer = imputer.fit(x[:, 3:4])
x[:, 3:4] = imputer.transform(x[:, 3:4])
x



 