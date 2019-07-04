# Data Preporcssing Template

# Importing the libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset:
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print(X)
print("")
#print(Y)

# Taking care of missing data:
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)