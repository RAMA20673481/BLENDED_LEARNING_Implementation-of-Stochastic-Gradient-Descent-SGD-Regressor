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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SGD Regressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # Default settings
sgd_model.fit(X_train, y_train)

# Predictions on test set
y_pred = sgd_model.predict(X_test)

# Evaluate the model
print("Model Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
Developed by:G.Ramanujam 
RegisterNumber: 212224240129
*/
```

## Output:
![Screenshot 2025-05-05 153014](https://github.com/user-attachments/assets/a6c2b21f-369e-4028-ba6f-5c33c46c3069)



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
