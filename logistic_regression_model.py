import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

df = pd.read_csv("heart_clean.csv")
print("Number of rows to learn from:", len(df))

# Drop cardio as that's what we are predicting, every other column is a feature
X = df.drop(columns=["cardio"])
y = df["cardio"].astype(int)

# List all features
print("\nFeatures:", list(X.columns))

# Show how balanced the dataset is
# for example, 0 = 0.550 1 = 0.450, means 55% of patients in dataset don't have heart disease
print("Class balance:")
print(y.value_counts(normalize=True).round(3))

# 80 / 20 split, random state so every model uses the same rows, stratify to keep the class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9, stratify=y
)
print("\nTrain rows:", len(X_train), "| Test rows:", len(X_test))


# Checking for leakage, so no training rows show in the test set
print("\nLeakage checks")

# No row index should be in both train and test
shared_index = set(X_train.index) & set(X_test.index)
print("Rows in both train and test (by index):", len(shared_index))

# Check how many test patients have an identical copy in train
identical_rows = X_test.merge(X_train.drop_duplicates(), how="inner")
print("Test rows with an identical copy in train:", len(identical_rows),
      f"({len(identical_rows) / len(X_test):.1%} of test set)")

# Make sure no feature is basically the answer
target_corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print("Highest correlation with cardio:", target_corr.index[0], round(target_corr.iloc[0], 3))
if target_corr.iloc[0] > 0.9:
    print("WARNING: a feature is almost identical to the target, possible leakage")
else:
    print("No feature is suspiciously close to the target")

# Scale features so height and weight don't dominate compared to smoking (0/1)
# Using a pipeline to scale ensures that there's no data leaks
# Model 1, logistic regression
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# Train the model
model.fit(X_train, y_train)

# Use the model to predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Prob. of cardio = 1

# Evaluate the model
print("\nResults")
# What % the model got right
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
# Chance that the model ranked someone who is sick, with a higher risk rate than someone who is healthy
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))

# Use a 10-fold CV to ensure the split didn't just get lucky
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
print("10-fold CV accuracy mean:", round(cv_scores.mean(), 4), "+/- st dev: ", round(cv_scores.std(), 4))

# Define the Logistic Regression pipeline
grid_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=3000))
])

# Define the parameters to test
param_grid = {
    "logreg__C": [0.01, 0.1, 0.5, 1, 2, 10, 100],
    "logreg__penalty": ["l1", "l2"],
    "logreg__solver": ["liblinear"]
}

# Set up 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=9
)

# Run Grid Search
grid_search = GridSearchCV(
    estimator=grid_model,
    param_grid=param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1
)

# Fit using training data only
grid_search.fit(X_train, y_train)

# Display the best settings
print("Best parameters:", grid_search.best_params_)
print("Best CV recall:", grid_search.best_score_)

# Train the final model using the best settings
final_tuned_model = grid_search.best_estimator_
joblib.dump(final_tuned_model, "model1_logistic_regression_tuned.joblib")

# Evaluate on the held-out test data
tuned_pred = final_tuned_model.predict(X_test)
tuned_prob = final_tuned_model.predict_proba(X_test)[:, 1]

# Compare train and test accuracy to check for over / underfitting
print("\nOver / underfitting check")
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
gap = train_acc - test_acc
print("Train accuracy:", round(train_acc, 4))
print("Test accuracy:", round(test_acc, 4))
print("Gap (train - test):", round(gap, 4))

if gap > 0.05:
    print("Verdict: overfitting, the model does much better on data it has seen")
elif train_acc < 0.75:
    print("Verdict: no overfitting, but train accuracy is low too so some underfitting")
else:
    print("Verdict: good fit, train and test are close")

# Learning curve, lines far apart = overfitting, both low = underfitting
sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring="accuracy",
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], n_jobs=-1,
)
plt.figure(figsize=(7, 5))
plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Training accuracy")
plt.plot(sizes, val_scores.mean(axis=1), "o-", label="Validation accuracy")
plt.xlabel("Number of training rows")
plt.ylabel("Accuracy")
plt.title("Learning curve - Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model1_learning_curve.png", dpi=120)
plt.show()


# Confusion matrix shows correct and wrong guesses for healthy and sick
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["Healthy", "Sick"], columns=["Pred Healthy", "Pred Sick"]))

# Visual version of the same matrix
ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Sick"]).plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("model1_confusion_matrix.png", dpi=120)
plt.show()

# Precision, recall and f1 for both classes
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Sick"]))

# Save the model
joblib.dump(model, "model1_logistic_regression.joblib")
print("Saved model1_logistic_regression.joblib")


# Check if dropping smoke and alco changes anything
drop_cols = ["smoke", "alco"]
model_small = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))])
model_small.fit(X_train.drop(columns=drop_cols), y_train)
pred_small = model_small.predict(X_test.drop(columns=drop_cols))
prob_small = model_small.predict_proba(X_test.drop(columns=drop_cols))[:, 1]
print("\nWith smoke + alco    -> Accuracy:", round(accuracy_score(y_test, y_pred), 4), "AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("Without smoke + alco -> Accuracy:", round(accuracy_score(y_test, pred_small), 4), "AUC:", round(roc_auc_score(y_test, prob_small), 4))


# BASELINE VS GRID SEARCH TUNED MODEL

baseline_report = classification_report(
    y_test, y_pred, output_dict=True
)

tuned_report = classification_report(
    y_test, tuned_pred, output_dict=True
)

comparison = pd.DataFrame({
    "Baseline": [
        accuracy_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob),
        baseline_report["1"]["precision"],
        baseline_report["1"]["recall"],
        baseline_report["1"]["f1-score"]
    ],
    "GridSearchCV Tuned": [
        accuracy_score(y_test, tuned_pred),
        roc_auc_score(y_test, tuned_prob),
        tuned_report["1"]["precision"],
        tuned_report["1"]["recall"],
        tuned_report["1"]["f1-score"]
    ]
}, index=[
    "Accuracy",
    "ROC AUC",
    "Sick Precision",
    "Sick Recall",
    "Sick F1-score"
])

print("\nBaseline vs GridSearchCV Logistic Regression")
print(comparison.round(4))

comparison.to_csv("logistic_regression_comparison.csv")
print("Comparison saved.")

""" Results

Class balance:
cardio
0    0.505
1    0.495

Good balance in dataset, no need for class weighting for imbalance

Train rows: 54867 | Test rows: 13717

Leakage checks
Rows in both train and test (by index): 0
Test rows with an identical copy in train: 335 (2.4% of test set)
Highest correlation with cardio: ap_hi 0.431
No feature is suspiciously close to the target

Results
Accuracy: 0.7259
ROC AUC: 0.7887

Accuracy and ROC AUC are okay, acceptable results, tuning may give a small improvement

10-fold CV accuracy mean: 0.7278 +/- st dev:  0.0061


Over / underfitting check
Train accuracy: 0.728
Test accuracy: 0.7259
Gap (train - test): 0.0021
Verdict: no overfitting, but train accuracy is low too so some underfitting

Confusion matrix:
         Pred Healthy  Pred Sick
Healthy          5453       1475
Sick             2284       4500

Sick Recall is the main weakness of this model, when looking at a sick patient there's 33% chance of a false negative,
potentially lower decision threshold, or change weights, false positive much more preferable

Classification report:
              precision    recall  f1-score   support

     Healthy       0.70      0.79      0.74      6928
        Sick       0.75      0.66      0.71      6784

    accuracy                           0.73     13712
   macro avg       0.73      0.73      0.72     13712
weighted avg       0.73      0.73      0.72     13712

As mentioned above sick recall needs improving, other than that decent accuracy

"""
