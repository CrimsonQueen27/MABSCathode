# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:00:45 2024

@author: merye
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# Function: Load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath, index_col=0)
    X = data.drop(columns=['y_true'])
    y = data['y_true']
    return train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Function: Standardize data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)

# Function: Train and evaluate XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test, param_grid):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.01,
        random_state=0,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=kfold,
        random_state=0,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    return best_model, train_preds, test_preds

# Function: Train and evaluate LightGBM model
def train_lightgbm(X_train, y_train, X_test, y_test):
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        boosting_type='gbdt',
        objective='regression',
        colsample_bytree=0.6,
        max_depth=9,
        min_child_samples=3,
        num_leaves=13,
        subsample=0.2,
        learning_rate=0.01,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        force_col_wise=True
    )
    lgb_model.fit(X_train, y_train)
    train_preds = lgb_model.predict(X_train)
    test_preds = lgb_model.predict(X_test)

    return lgb_model, train_preds, test_preds

# Function: Train and evaluate ANN model
def train_ann(X_train, y_train, X_test, y_test):
    ann_model = MLPRegressor(
        hidden_layer_sizes=(150, 100, 40),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate_init=0.01,
        max_iter=100,
        batch_size=64,
        random_state=42,
        verbose=True
    )
    ann_model.fit(X_train, y_train)
    train_preds = ann_model.predict(X_train)
    test_preds = ann_model.predict(X_test)

    return ann_model, train_preds, test_preds

# Function: Evaluate model performance
def evaluate_performance(y_train, train_preds, y_test, test_preds):
    metrics = {
        "Train R2": r2_score(y_train, train_preds),
        "Test R2": r2_score(y_test, test_preds),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, train_preds)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, test_preds))
    }
    return metrics

# Function: Plot predictions
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title(f'{title} - Predicted vs Actual')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.grid()
    plt.show()

# Function: Compare model metrics
def compare_metrics(metrics_dict):
    models = list(metrics_dict.keys())
    r2_scores = [metrics["Test R2"] for metrics in metrics_dict.values()]
    rmse_scores = [metrics["Test RMSE"] for metrics in metrics_dict.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width/2, r2_scores, width, label='R² Score')
    bar2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE')

    ax.set_xlabel('Models')
    ax.set_title('Comparison of R² and RMSE Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    filepath = r"C:\Users\merye\OneDrive\Masaüstü\Tez1\Datas\0.63VRHE_data_after_clearning.xlsx"
   
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(filepath)
    X_train_stan, X_test_stan = standardize_data(X_train, X_test)

    # Define hyperparameter grid for XGBoost
    param_grid = {
        'max_depth': [4, 5, 6],
        'min_child_weight': [2, 3],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8],
        'reg_alpha': [1],
        'reg_lambda': [1]
    }

    # Train models
    xgb_model, xgb_train_preds, xgb_test_preds = train_xgboost(X_train_stan, y_train, X_test_stan, y_test, param_grid)
    lgb_model, lgb_train_preds, lgb_test_preds = train_lightgbm(X_train_stan, y_train, X_test_stan, y_test)
    ann_model, ann_train_preds, ann_test_preds = train_ann(X_train_stan, y_train, X_test_stan, y_test)

    # Evaluate models
    xgb_metrics = evaluate_performance(y_train, xgb_train_preds, y_test, xgb_test_preds)
    lgb_metrics = evaluate_performance(y_train, lgb_train_preds, y_test, lgb_test_preds)
    ann_metrics = evaluate_performance(y_train, ann_train_preds, y_test, ann_test_preds)

    # Print metrics
    print("XGBoost Metrics:", xgb_metrics)
    print("LightGBM Metrics:", lgb_metrics)
    print("ANN Metrics:", ann_metrics)

    # Visualize predictions
    plot_predictions(y_test, xgb_test_preds, "XGBoost")
    plot_predictions(y_test, lgb_test_preds, "LightGBM")
    plot_predictions(y_test, ann_test_preds, "ANN")

    # Compare metrics in a single graph
    all_metrics = {
        "XGBoost": xgb_metrics,
        "LightGBM": lgb_metrics,
        "ANN": ann_metrics
    }
    compare_metrics(all_metrics)