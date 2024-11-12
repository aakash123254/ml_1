# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:02:44 2024

@author: srush
"""

# TENNIS EXAMPLE 
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


play_tennis = pd.read_csv("D://MCA//ML//tennisgame.csv")
play_tennis.head()

number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])
play_tennis

#define the features and the target variables
#features = play_tennis.iloc[:, :-1].values
#target = play_tennis.iloc[:, -1].values
#features
#target


#define the features and the target variables
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

features_train, features_test, target_train, target_test = train_test_split(play_tennis[features],play_tennis[target],test_size = 0.33,random_state = 54)


model = GaussianNB()
model.fit(features_train, target_train) 
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(accuracy)

print (model.predict([[1,2,0,1]]))
print(model.predict([[2,0,0,0]]))
print(model.predict([[0,0,0,1]]))
