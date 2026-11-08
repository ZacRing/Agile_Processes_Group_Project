import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

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

# Splitting the train and test sets in an 80 / 20 split, random state so when making other models we can fairly compare them, stratify so train and test have same class balance as data set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9, stratify=y
)
print("\nTrain rows:", len(X_train), "| Test rows:", len(X_test))

# Scale features so height and weight (often 50/100+) don't dominate compared to smoking (0/1)
# Using a pipeline to scale ensures that there's no data leaks
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

# Train the model
model.fit(X_train, y_train)

# Use the model to predict based on the training
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Prob. of cardio = 1

# Evaluate the model
print("\nResults")
# What % the model got right
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
# Chance that the model ranked someone who is sick, with a higher risk rate than someone who is healthy
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))

# Use a 5-fold CV to ensure the train and test split didn't just get lucky
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("5-fold CV accuracy mean:", round(cv_scores.mean(), 4), "+/- st dev: ", round(cv_scores.std(), 4))

# Use a confusion matrix to show false positives, correct positives, false negatives, correct negatives
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["Healthy", "Sick"], columns=["Pred Healthy", "Pred Sick"]))

# Classification report gives a view of precision, recall. f1 score for both classes and the average
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Sick"]))

joblib.dump(model, "model1_logistic_regression.joblib")
print("Saved model1_logistic_regression.joblib")


""" Results

Class balance:
cardio
0    0.505
1    0.495

Good balance in dataset, no need for class weighting for imbalance

Train rows: 54867 | Test rows: 13717

Results
Accuracy: 0.7271
ROC AUC: 0.7912

Accuracy and ROC AUC are okay, acceptable results, tuning may give a small improvement

5-fold CV accuracy mean: 0.7274 +/- st dev:  0.0054

Very similar to test accuracy, model performs consistently on unseen data

Confusion matrix:
         Pred Healthy  Pred Sick
Healthy          5411       1519
Sick             2224       4563

Sick Recall is the main weakness of this model, when looking at a sick patient there's 33% chance of a false negative, 
potentially lower decision threshold, or change weights, false positive much more preferable

Classification report:
              precision    recall  f1-score   support

     Healthy       0.71      0.78      0.74      6930
        Sick       0.75      0.67      0.71      6787

    accuracy                           0.73     13717
   macro avg       0.73      0.73      0.73     13717
weighted avg       0.73      0.73      0.73     13717

As mentioned above sick recall needs improving, other than that decent accuracy

"""