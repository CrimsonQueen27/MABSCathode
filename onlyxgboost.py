# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:55:35 2024

@author: merye
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load and prepare the dataset
data_final = pd.read_excel(r"C:\Users\merye\OneDrive\Masaüstü\Tez1\0.8VRHE_data_without_formulas_updated.xlsx", index_col=0)

X = data_final.drop(columns=['y_true'])
y = data_final['y_true']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Convert to numpy arrays for compatibility
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train)
y_test_np = np.array(y_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

# Define the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,  # Number of trees
    learning_rate=0.01,  # Learning rate
    random_state=0,
    n_jobs=-1  # Utilize all CPU cores
)

# Define the hyperparameter grid
param_grid = {
    'max_depth': [4, 5, 6],           # Maximum depth of trees
    'min_child_weight': [2, 3],       # Minimum child weight
    'subsample': [0.8, 1],            # Subsampling ratio
    'colsample_bytree': [0.8],        # Feature sampling ratio
    'reg_alpha': [1],                 # L1 regularization
    'reg_lambda': [1]                 # L2 regularization
}

# Set up K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=kfold,
    scoring='r2',  # Use R2 as the evaluation metric
    n_jobs=-1  # Utilize all CPU cores
)

# Fit the model
grid_search.fit(X_train_scaled, y_train_np)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best R2 Score (Cross-Validation):", grid_search.best_score_)

# Predict on training and test sets
train_predictions = grid_search.best_estimator_.predict(X_train_scaled)
test_predictions = grid_search.best_estimator_.predict(X_test_scaled)

# Evaluate model performance
print("R2 Score (Train):", r2_score(y_train_np, train_predictions))
print("R2 Score (Test):", r2_score(y_test_np, test_predictions))
print("MSE (Train):", mean_squared_error(y_train_np, train_predictions))
print("MSE (Test):", mean_squared_error(y_test_np, test_predictions))

# Optional: Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_np, test_predictions, alpha=0.7, label='Predictions')
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], '--r', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')
plt.legend()
plt.show()
