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
#print(dataset)

################################################################################
# Independent Variable (# of years of experience)(X)
################################################################################
#X = dataset.iloc[:, 0].values
X = dataset.iloc[:, :-1].values
#print("")
#print(X)

################################################################################
# Dependent Variable (Salary)(Y)
################################################################################
Y = dataset.iloc[:, 1].values
#print("")
#print(Y)

################################################################################
# Splitting the dataset into the Training datset and the Test dataset:
################################################################################
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

################################################################################
# Fitting Simple Linear Regression to the Training Set:
################################################################################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

################################################################################
# Predicting the Test Set Results:
################################################################################
Y_pred = regressor.predict(X_test)
print(Y_pred)

################################################################################
# Visualizing the Training Set Results:
################################################################################
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Expereince')
plt.ylabel('Salary')
plt.show()

################################################################################
# Visualizing the Test Set Results:
################################################################################
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Expereince')
plt.ylabel('Salary')
plt.show()




