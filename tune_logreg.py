import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

# 1. Load your original model
old_model = joblib.load("model1_logistic_regression.joblib")
print(old_model)  # shows its current settings

df = pd.read_csv("heart_clean.csv")
print("Number of rows to learn from:", len(df))

# Drop cardio as that's what we are predicting, every other column is a feature
X = df.drop(columns=["cardio"])
y = df["cardio"].astype(int)

# 2. Same split for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=9
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=9))
])

# Define the hyperparameter grid
param_grid = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']
}

# Run Grid Search with 5-fold cross-validation
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best Cross-Validation Score: {grid.best_score_:.4f}")