# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:00:53 2024

@author: merye
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Load dataset
data_loop_2_final = pd.read_excel(r"C:\Users\merye\OneDrive\Masaüstü\Tez1\Datas\0.63VRHE_data_after_clearning.xlsx", index_col=0)

# Split the dataset into features and target
X = data_loop_2_final.drop(columns=['y_true'])
y = data_loop_2_final['y_true']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Standardize the features
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
X_test_stan = scaler.transform(X_test)

# Initialize models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=0),
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.01, random_state=0)
}

# Initialize KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# Dictionary to store results
model_scores = {'Model': [], 'R² Score (Train)': [], 'R² Score (Test)': [], 'MSE (Train)': [], 'MSE (Test)': []}

# Train and evaluate models
for model_name, model in models.items():
    # Train the model on the training data
    model.fit(X_train_stan, y_train)
    
    # Make predictions on the train and test set
    train_pred = model.predict(X_train_stan)
    test_pred = model.predict(X_test_stan)
    
    # Calculate R² and MSE for training and testing data
    r2_train = r2_score(y_train, train_pred)
    r2_test = r2_score(y_test, test_pred)
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)
    
    # Append the results to the dictionary
    model_scores['Model'].append(model_name)
    model_scores['R² Score (Train)'].append(r2_train)
    model_scores['R² Score (Test)'].append(r2_test)
    model_scores['MSE (Train)'].append(mse_train)
    model_scores['MSE (Test)'].append(mse_test)

# Convert the dictionary to a dataframe
results_df = pd.DataFrame(model_scores)

# Print results for inspection
print(results_df)

# Plot comparison of models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot R² scores
sns.barplot(x='Model', y='R² Score (Test)', data=results_df, ax=axes[0], palette='Set1')
axes[0].set_title('Comparison of Models (R² Score on Test Data)')
axes[0].set_ylabel('R² Score')
axes[0].grid(True)  # Add grid to the first plot

# Add values on top of bars for R² scores
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.2f', label_type='edge', padding=3)

# Plot MSE scores
sns.barplot(x='Model', y='MSE (Test)', data=results_df, ax=axes[1], palette='Set2')
axes[1].set_title('Comparison of Models (MSE on Test Data)')
axes[1].set_ylabel('MSE')
axes[1].grid(True)  # Add grid to the second plot

# Add values on top of bars for MSE scores
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.2f', label_type='edge', padding=3)

plt.tight_layout()
plt.show()
