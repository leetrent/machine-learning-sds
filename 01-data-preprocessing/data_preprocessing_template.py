################################################################################
# Data Preporcssing Template
################################################################################

################################################################################
# Importing the libraries:
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

################################################################################
# Importing the dataset:
################################################################################
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

################################################################################
# DEBUG: Importing the dataset:
################################################################################
#print("---------------------------------------------------------")
#print("------------------------ X ------------------------------")
#print("---------------------------------------------------------")
#print(X)
#print("---------------------------------------------------------")
#print("------------------------ Y ------------------------------")
#print("---------------------------------------------------------")
#print(Y)
#print("---------------------------------------------------------")

################################################################################
# Taking care of missing data:
################################################################################
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

################################################################################
# DEBUG: Taking care of missing data:
################################################################################
#print("---------------------------------------------------------")
#print("------------------------ X ------------------------------")
#print("---------------------------------------------------------")
#print(X)

################################################################################
# Encoding categorical data:
################################################################################
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

################################################################################
# Encoding independent variable 'Country':
################################################################################
col_trans = make_column_transformer((OneHotEncoder(),[0]), remainder="passthrough")
X = col_trans.fit_transform(X)

################################################################################
# Converts object X to int32 so it can be visible in Variable Explorer in Spyder
################################################################################
#X = X.astype(np.int32)
X[:,0:3] = X[:, 0:3].astype(np.int32)

################################################################################
# DEBUG: Encoding independent variable 'Country':
################################################################################
#print("---------------------------------------------------------")
#print("------------------------ X ------------------------------")
#print("---------------------------------------------------------")
#print(X)
#print("---------------------------------------------------------")

################################################################################
# Encoding dependent variable 'Purchased':
################################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

################################################################################
# DEBUG: dependent variable 'Purchased'
################################################################################
#print("---------------------------------------------------------")
#print("------------------------ Y ------------------------------")
#print("---------------------------------------------------------")
#print(Y)
#print("---------------------------------------------------------")

################################################################################
# Splitting the dataset into the Training datset and the Test dataset:
################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

################################################################################
# DEBUG: 'X_train'
################################################################################
#print("---------------------------------------------------------")
#print("------------------ X_train ------------------------------")
#print("---------------------------------------------------------")
#print(X_train)

################################################################################
# DEBUG: 'X_test'
################################################################################
print("---------------------------------------------------------")
print("------------------- X_test ------------------------------")
print("---------------------------------------------------------")
print(X_test)

################################################################################
# DEBUG: 'Y_train'
################################################################################
#print("---------------------------------------------------------")
#print("------------------ Y_train ------------------------------")
#print("---------------------------------------------------------")
#print(Y_train)

################################################################################
# DEBUG: 'X_test'
################################################################################
#print("---------------------------------------------------------")
#print("------------------- Y_test ------------------------------")
#print("---------------------------------------------------------")
#print(Y_test)

################################################################################
# Feature Scaling:
################################################################################
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[, 2:3] = sc_X.fit_transform(X_train[, 2:3])
X_test[, 2:3] = sc_X.transform(X_test[, 2:3])

################################################################################
# DEBUG: 'X_train'
################################################################################
#print("---------------------------------------------------------")
#print("------------------ X_train ------------------------------")
#print("---------------------------------------------------------")
#print(X_train)

################################################################################
# DEBUG: 'X_test'
################################################################################
print("---------------------------------------------------------")
print("------------------- X_test ------------------------------")
print("---------------------------------------------------------")
print(X_test)
print("---------------------------------------------------------")
print("")




