import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
play_tennis = pd.read_csv("C:/Users/AAKASH/OneDrive/Documents/JG/SEM - 3/ML/MachineLearning//PlayTennis.csv")
print(play_tennis.head())

# Print column names to verify
print(play_tennis.columns)

# Strip leading/trailing spaces from column names
play_tennis.columns = play_tennis.columns.str.strip()

# Label encoding
number = LabelEncoder()
play_tennis['Outlook'] = number.fit_transform(play_tennis['Outlook'])
play_tennis['Temperature'] = number.fit_transform(play_tennis['Temperature'])
play_tennis['Humidity'] = number.fit_transform(play_tennis['Humidity'])
play_tennis['Wind'] = number.fit_transform(play_tennis['Wind'])
play_tennis['Play Tennis'] = number.fit_transform(play_tennis['Play Tennis'])

# Define features and target
features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play Tennis"

# Split dataset
features_train, features_test, target_train, target_test = train_test_split(play_tennis[features], play_tennis[target], test_size=0.33, random_state=54)

# Train Naive Bayes model
model = GaussianNB()
model.fit(features_train, target_train)

# Make predictions
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print("Accuracy:", accuracy)

# Test predictions with new data
print(model.predict(pd.DataFrame([[1, 2, 0, 1]], columns=features)))
print(model.predict(pd.DataFrame([[2, 0, 0, 0]], columns=features)))
print(model.predict(pd.DataFrame([[0, 0, 0, 1]], columns=features)))

# input_data_1 = pd.DataFrame([[1, 2, 0, 1]], columns=features)  # Make sure the input is a DataFrame with correct feature names
# input_data_2 = pd.DataFrame([[2, 0, 0, 0]], columns=features)
# input_data_3 = pd.DataFrame([[0, 0, 0, 1]], columns=features)
#
# print(model.predict(input_data_1))
# print(model.predict(input_data_2))
# print(model.predict(input_data_3))