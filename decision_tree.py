import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# model 2 decision tree tested the exact same way as marks logistic regression so we can compare them fairly

df = pd.read_csv("heart_clean.csv")
print("Number of rows to learn from:", len(df))

# cardio is what we are predicting every other column is a feature
X = df.drop(columns=["cardio"])
y = df["cardio"].astype(int)

print("\nFeatures:", list(X.columns))
print("Class balance:")
print(y.value_counts(normalize=True).round(3))

# same split as mark 80 20 with random state 9 and stratify so all 3 models get the exact same rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9, stratify=y
)
print("\nTrain rows:", len(X_train), "| Test rows:", len(X_test))

# no scaling here cos tree models only ask yes or no questions like is ap_hi over 130 so it doesnt matter that height is in the 100s and smoke is 0 or 1

# model 2 decision tree asks yes or no questions to split patients into smaller and smaller groups
model = DecisionTreeClassifier(random_state=9)

# train the model
model.fit(X_train, y_train)

# use the model to predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # chance of cardio being 1

# evaluate the model
print("\nResults")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))

# 5 fold cv to make sure the split didnt just get lucky
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print("5 fold CV accuracy mean:", round(cv_scores.mean(), 4), "+/- st dev: ", round(cv_scores.std(), 4))

# confusion matrix shows correct and wrong guesses for healthy and sick
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["Healthy", "Sick"], columns=["Pred Healthy", "Pred Sick"]))

# precision recall and f1 for both classes
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Sick"]))

# save the model so we dont have to retrain it every time
joblib.dump(model, "model2_decision_tree.joblib", compress=3)
print("Saved model2_decision_tree.joblib")

""" Results

Class balance:
cardio
0    0.505
1    0.495

Good balance in dataset, no need for class weighting for imbalance

Train rows: 54867 | Test rows: 13717

Results
Accuracy: 0.632
ROC AUC: 0.6321


Accuracy and ROC AUC are poor, tuning may give a small improvement

5 fold CV accuracy mean: 0.6252 +/- st dev:  0.0028

Very similar to test accuracy, model performs consistently on unseen data

Confusion matrix:
         Pred Healthy  Pred Sick
Healthy          4364       2566
Sick             2482       4305

Recall is the main weakness of this model, when looking at a patient there's 37% chance of a wrong result, 
potentially lower decision threshold, or change weights, false positive much more preferable

Classification report:
              precision    recall  f1-score   support

     Healthy       0.64      0.63      0.63      6930
        Sick       0.63      0.63      0.63      6787

    accuracy                           0.63     13717
   macro avg       0.63      0.63      0.63     13717
weighted avg       0.63      0.63      0.63     13717

Recall and accuracy need improving, poor overall results

"""