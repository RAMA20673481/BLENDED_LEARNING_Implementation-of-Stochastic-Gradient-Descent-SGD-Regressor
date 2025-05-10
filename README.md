# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Libraries:
  Use pandas for data manipulation and scikit-learn for modeling and evaluation tasks.

2.Dataset Loading:
  Load the dataset from encoded_car_data.csv.

3.Feature and Target Selection:
  Identify the features (X) and target variable (y) for modeling.

4.Data Splitting:
  Split the data into training and testing sets, maintaining an 80-20 split.

5.Model Training:
  Train a Stochastic Gradient Descent (SGD) Regressor using the training dataset.

6.Prediction Generation:
  Use the trained model to predict car prices on the test data.

7.Model Evaluation:
  Assess the model's performance using metrics such as Mean Squared Error (MSE) and R² score.

8.Output Coefficients:
  Display the model’s coefficients and intercept. 
```
## Program:
```
/*
# Program to implement SGD Regressor for linear regression.
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print("\n\n")
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
x = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1, 1)).ravel()


# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd_model.fit(x_train, y_train)

# Making predictions
y_pred = sgd_model.predict(x_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print()
print()
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\n\n")
print("Model Coefficients")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
print("\n\n")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()

Developed by:G.Ramanujam 
RegisterNumber: 212224240129
*/
```

## Output:

## LOAD THE DATASET:
![Screenshot 2025-05-10 160523](https://github.com/user-attachments/assets/196d5f7c-f808-4833-a5dd-bbbab46b2852)
![Screenshot 2025-05-10 160634](https://github.com/user-attachments/assets/2733d58b-47f1-49be-863f-7d289dababc3)
## EVALUATION METRICS:
![Screenshot 2025-05-10 160824](https://github.com/user-attachments/assets/f71e2f93-40f3-4d16-b980-0996c919815d)
## MODEL COEFFICIENTS:
![Screenshot 2025-05-10 160836](https://github.com/user-attachments/assets/df456305-dbb7-411c-a05f-9dc8d2888783)
## VISUALIZATION OF ACTUAL VS PREDICTED VALUES:
![Screenshot 2025-05-10 161026](https://github.com/user-attachments/assets/873ec71c-7190-413b-9b1c-01b3a15a3c75)


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
