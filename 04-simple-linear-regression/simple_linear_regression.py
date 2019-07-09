################################################################################
# Simple Linear Regression
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
dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

################################################################################
# Independent Variable (# of years of experience)(X)
################################################################################
X = dataset.iloc[:, 0].values
#X = dataset.iloc[:, -1].values
print("")
print(X)

################################################################################
# Dependent Variable (Salary)(Y)
################################################################################
Y = dataset.iloc[:, 1].values
print("")
print(Y)

################################################################################
# Splitting the dataset into the Training datset and the Test dataset:
################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
