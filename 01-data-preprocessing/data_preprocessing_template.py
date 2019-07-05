# Data Preporcssing Template

# Importing the libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset:
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
#print(X)
#print("")
#print(Y)

# Taking care of missing data:
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)

# Encoding categorical data:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#print("")
#print (X)

# Encode independent variable 'Country':

onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
print("")
print(X)

# Encode dependent variable 'Purchased':
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)