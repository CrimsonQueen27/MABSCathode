# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:35:10 2024

@author: merye
"""
import numpy as np
import pandas as pd

from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

# Step 1: Load Data
data_loop_2_final = pd.read_excel(r"C:\Users\merye\OneDrive\Masaüstü\Tez1\0.63VRHE_data_after_clearning.xlsx", index_col=0)

# Separate features and target
X_train_updat_pd = data_loop_2_final.drop(columns=['y_true'])
Y_train_updat_pd = data_loop_2_final['y_true']

# Step 2: Convert formulas to Composition objects
X_train_updat_pd = StrToComposition().featurize_dataframe(X_train_updat_pd, 'formula')

# Step 3: Define Featurizers
featurizers = [
    ElementProperty.from_preset("magpie"),  # Elemental properties
    Stoichiometry(),  # Stoichiometric features
    ValenceOrbital()  # Features based on valence orbitals
]

# Combine featurizers into a MultipleFeaturizer
multi_featurizer = MultipleFeaturizer(featurizers)

# Step 4: Apply Featurizers
X_train_updat_pd = multi_featurizer.featurize_dataframe(X_train_updat_pd, col_id='composition', ignore_errors=True)

# Drop unnecessary columns (e.g., original formula and composition objects if not needed later)
X_train_updat_pd = X_train_updat_pd.drop(columns=['composition', 'formula'])

# Step 5: Proceed with Standardization and Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X_train_updat_pd, Y_train_updat_pd, test_size=0.2, random_state=0, shuffle=True
)

scaler_ = StandardScaler()
sca_fit_ = scaler_.fit(X_train)
X_train_stan_ = sca_fit_.transform(X_train)
X_test_stan_ = sca_fit_.transform(X_test)

# Step 6: Train XGBoost Model (as in your original code)
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV

# KFold Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

# Hyperparameter Tuning
param_grid = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [2, 3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8],
    'reg_alpha': [1],
    'reg_lambda': [1],
}

xgb_test = XGBRegressor(
    n_estimators=100,
    learning_rate=0.01,
    random_state=0,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    xgb_test,
    param_distributions=param_grid,
    n_iter=20,
    cv=kfold,
    random_state=0,
    n_jobs=-1
)

random_search.fit(X_train_stan_, y_train)
print(random_search.best_params_)
print(random_search.best_score_)
