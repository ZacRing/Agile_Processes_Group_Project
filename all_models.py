import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
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
in_train = X_test.merge(X_train.drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").values
print("Test rows with an identical copy in train:", in_train.sum(),
      f"({in_train.mean():.1%} of test set)")

# Make sure no feature is basically the answer
target_corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
print("Highest correlation with cardio:", target_corr.index[0], round(target_corr.iloc[0], 3))
if target_corr.iloc[0] > 0.9:
    print("WARNING: a feature is almost identical to the target, possible leakage")
else:
    print("No feature is suspiciously close to the target")


# Scale features so height and weight don't dominate compared to smoking (0/1)
# Using a pipeline to scale ensures that there's no data leaks
models = {
    # Model 1, logistic regression
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
    # Model 2, decision tree
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(random_state=9)),
    ]),
    # Model 3, random forest
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=9, n_jobs=-1)),
    ]),
}

# Filenames for saving each model
save_names = {
    "Logistic Regression": "model1_logistic_regression.joblib",
    "Decision Tree": "model2_decision_tree.joblib",
    "Random Forest": "model3_random_forest.joblib",
}

# Store results to compare at the end
results = {}

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train the model
    model.fit(X_train, y_train)

    # Use the model to predict on the test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Prob. of cardio = 1

    # Evaluate the model
    # What % the model got right
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    # Chance that the model ranked someone who is sick, with a higher risk rate than someone who is healthy
    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", round(auc, 4))

    # Use a 10-fold CV to ensure the split didn't just get lucky
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    print("10-fold CV accuracy mean:", round(cv_scores.mean(), 4), "+/- st dev: ", round(cv_scores.std(), 4))

    # Accuracy without the copied rows, should barely change
    acc_unique = accuracy_score(y_test.values[~in_train], y_pred[~in_train])
    print("Accuracy without copied rows:", round(acc_unique, 4))

    # Compare train and test accuracy to check for over / underfitting
    train_acc = accuracy_score(y_train, model.predict(X_train))
    gap = train_acc - acc
    print("Train accuracy:", round(train_acc, 4), "| Gap (train - test):", round(gap, 4))

    # Confusion matrix shows correct and wrong guesses for healthy and sick
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(pd.DataFrame(cm, index=["Healthy", "Sick"], columns=["Pred Healthy", "Pred Sick"]))

    # Precision, recall and f1 for both classes
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Sick"]))

    # Learning curve, lines far apart = overfitting, both low = underfitting
    sizes, lc_train, lc_val = learning_curve(
        model, X_train, y_train, cv=10, scoring="accuracy",
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    )

    # Precision, recall and f1 are for the sick class
    results[name] = {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "roc_auc": auc,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_acc": train_acc,
        "gap": gap,
        "acc_unique": acc_unique,
        "cm": cm,
        "y_prob": y_prob,
        "lc": (sizes, lc_train, lc_val),
    }

    # Save the model
    joblib.dump(model, save_names[name], compress=3)
    print("Saved", save_names[name])


# Visuals
names = list(results.keys())

# Confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
max_count = max(r["cm"].max() for r in results.values())
for ax, name in zip(axes, names):
    disp = ConfusionMatrixDisplay(results[name]["cm"], display_labels=["Healthy", "Sick"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    disp.im_.set_clim(0, max_count)
    ax.set_title(name)
fig.suptitle("Confusion matrices")
plt.tight_layout()
plt.savefig("compare_confusion_matrices.png", dpi=120)
plt.show()

# Learning curves side by side, shading is the st dev across folds
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, name in zip(axes, names):
    sizes, lc_train, lc_val = results[name]["lc"]
    for scores, label in [(lc_train, "Training accuracy"), (lc_val, "Validation accuracy")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", label=label)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.2)
    ax.set_title(name)
    ax.set_xlabel("Number of training rows")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Accuracy")
axes[0].legend(loc="lower right")
fig.suptitle("Learning curves")
plt.tight_layout()
plt.savefig("compare_learning_curves.png", dpi=120)
plt.show()

# Bar chart comparing every model
metrics = ["accuracy", "cv_mean", "roc_auc", "precision", "recall", "f1"]
labels = ["Accuracy", "10-fold CV", "ROC AUC", "Precision (sick)", "Recall (sick)", "F1 (sick)"]
summary = pd.DataFrame({name: [results[name][m] for m in metrics] for name in names}, index=labels)
ax = summary.plot(kind="bar", figsize=(11, 6), width=0.8)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8)
ax.set_ylim(0.5, 0.85)
ax.set_ylabel("Score")
ax.set_title("Final model comparison")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("compare_final_metrics.png", dpi=120)
plt.show()


# Final table with all the results
print("\n" + "=" * 60)
print("Final comparison")
print("=" * 60)
table = pd.DataFrame({
    name: {
        "Accuracy": results[name]["accuracy"],
        "10-fold CV mean": results[name]["cv_mean"],
        "10-fold CV st dev": results[name]["cv_std"],
        "ROC AUC": results[name]["roc_auc"],
        "Precision (sick)": results[name]["precision"],
        "Recall (sick)": results[name]["recall"],
        "F1 (sick)": results[name]["f1"],
        "Train accuracy": results[name]["train_acc"],
        "Gap (train - test)": results[name]["gap"],
    }
    for name in names
})
print(table.round(4))