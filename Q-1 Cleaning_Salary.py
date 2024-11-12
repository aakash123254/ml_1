import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("RawData.csv")

# Display the columns and their count
print(data.columns)
print(len(data.columns))
print(len(data))

# Check data types and for any null values
print(data.dtypes)
print(data.isnull().values.any())
print("Total empty cells by column:", data.isnull().sum())

# Unique values in 'Location' and 'Salary'
print("Number of Unique Locations: ", len(data['Location'].unique()))
print("Number of Unique Salaries: ", len(data['Salary'].unique()))
print("Length of Salary data:",len(data['Salary'].unique()))
print("Unique Salaries:", data['Salary'].unique())

# Clean the 'Experience' column
exp = list(data.Experience)
min_ex = []
max_ex = []

for i in range(len(exp)):
    exp[i] = exp[i].replace("yrs", "").strip()
    min_ex.append(int(exp[i].split("-")[0].strip()))
    max_ex.append(int(exp[i].split("-")[1].strip()))

print("min_ex",min_ex)
print("max_ex",max_ex)

# Attaching the new experiences to the original dataset
data["minimum_exp"] = min_ex
data["maximum_exp"] = max_ex

# Label encoding for 'Location' and 'Salary'
le = LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])
data['Salary'] = le.fit_transform(data['Salary'])

# Prepare the final DataFrame
Index = data['Index']
Company = data['Company']
Location = data['Location']
Salary = data['Salary']
minimum_exp = data['minimum_exp']
maximum_exp = data['maximum_exp']

# Create a dictionary of lists for the DataFrame
dict_data = {
    'Index': Index,
    'Company': Company,
    'Location': Location,
    'Salary': Salary,
    'minimum_exp': minimum_exp,
    'maximum_exp': maximum_exp
}

# Create and save the DataFrame to a CSV file
df = pd.DataFrame(dict_data)



df.to_csv("File5.csv")