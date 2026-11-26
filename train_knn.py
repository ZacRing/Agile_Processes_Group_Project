"""
Build the KNN "similar patients" index for the website.

Put this next to logistic_regression_model.py and heart_clean.csv, then run:
    python train_knn.py

It saves knn_index.joblib in the same folder as
model1_logistic_regression_tuned.joblib (the project root), which is where
views.py already looks for the model.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart_clean.csv")

# Same features and same split as logistic_regression_model.py,
# so the neighbours come from the same rows the model learned from
X = df.drop(columns=["cardio"])
y = df["cardio"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=9, stratify=y
)

# Scale first so height (~165) and ap_hi (~120) don't drown out smoke (0/1)
# in the distance calculation, same reason the LR pipeline scales
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", NearestNeighbors(n_neighbors=11)),
])
knn.fit(X_train)

# Quick check that KNN agrees with the data: of each test patient's 5
# neighbours, does the majority vote match their real cardio label?
_, idx = knn["knn"].kneighbors(knn["scaler"].transform(X_test))
vote = (y_train.values[idx].mean(axis=1) >= 0.5).astype(int)
print("5-NN majority vote accuracy on test set:", round((vote == y_test.values).mean(), 4))

joblib.dump(
    {
        "pipeline": knn,
        "features": list(X.columns),
        "rows": X_train.reset_index(drop=True),   # raw values, for showing on the page
        "labels": y_train.reset_index(drop=True),
    },
    "knn_index.joblib",
    compress=3,
)
print("Saved knn_index.joblib with", len(X_train), "patients")