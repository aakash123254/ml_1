import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#load data set
df=pd.read_csv('Shirtsize.csv', sep='\t')

# Display the DataFrame columns and first few rows for inspection
print("Columns in DataFrame:", df.columns)

# Encode the target variable
le=LabelEncoder()
df['T Shirt Size']=le.fit_transform(df['T Shirt Size'])


# Select features and target variable
X = df.iloc[:, :-1].values  # Assuming all columns except the last are features
Y = df['T Shirt Size'].values
# Y = df.iloc[:,2].values

#Split traing and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)# n_neighbors means k
knn.fit(X_train, Y_train)

# Make predictions
prediction = knn.predict(X_test)

# Evaluate the model
print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, Y_test)*100))
print("Model Score",knn.score(X_test,Y_test))

# Confusion matrix
result = confusion_matrix(Y_test, prediction)
print("Confusion Matrix:")
print(result)

# classification report
result1 = classification_report(Y_test, prediction)
print("Classification Report:",)
print (result1)

result2 = accuracy_score(Y_test,prediction)
print("Accuracy:",result2)




