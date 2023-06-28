# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:06:52 2023

@author: liraz
"""


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings
import pickle
import madlan_data_prep 

# Disable ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


data = pd.read_excel('output_all_students_Train_v10.xlsx')
data = madlan_data_prep.prepare_data(data)


X = data.drop(['price'], axis=1)
y = data['price']

numerical_columns = ['room_number', 'Area']
categorical_columns = ['City', 'type', 'condition ']
    
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)],remainder='passthrough')
    

# Check for duplicate columns in X
#assert len(X.columns) == len(set(X.columns)), "Duplicate columns found in X"

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ElasticNet(max_iter=10000))])

# Define the grid of hyperparameters to search over
param_grid = {
    'classifier__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

cv = KFold(n_splits=10)

# Define the grid search object
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error')

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)

# Use the best model 
best_model = grid_search.best_estimator_

# Get the coefficients
coef = best_model.named_steps['classifier'].coef_
columns = X.columns

# Print the coefficients
for feature, weight in zip(columns, coef):
    print(f"{feature}: {weight}")

# Predict
y_pred = best_model.predict(X_test)
y_pred = y_pred.round(0)

#  MSE
mse = mean_squared_error(y_test, y_pred.round(0))
print('Mean Squared Error:', mse)

#  RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=10)
print('Cross-validation scores:', cv_scores)

# Create a dataframe with actual, predicted, and absolute difference
results_df = pd.DataFrame({
    'y_actual': y_test,
    'y_pred': y_pred,
    'abs_diff': abs(y_test - y_pred)
})

results_df

pickle.dump(best_model, open('trained_model.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))