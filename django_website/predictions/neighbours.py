"""
Find the 5 patients in the training data most similar to the person
who filled in the form. Put this file in the same app folder as views.py.
"""
from pathlib import Path

import joblib
import pandas as pd

# Same location views.py uses for the logistic regression model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNN_PATH = PROJECT_ROOT / "knn_index.joblib"

# Load once when the server starts, not on every request
knn_index = joblib.load(KNN_PATH)

LEVELS = {1: "Normal", 2: "Above normal", 3: "Well above normal"}


def find_neighbours(input_data, k=5):
    """input_data is the same dict views.py builds for the model."""
    features = knn_index["features"]
    pipeline = knn_index["pipeline"]

    person = pd.DataFrame([input_data], columns=features)
    distances, positions = pipeline["knn"].kneighbors(
        pipeline["scaler"].transform(person), n_neighbors=k
    )

    neighbours = []
    for distance, pos in zip(distances[0], positions[0]):
        row = knn_index["rows"].iloc[pos]
        neighbours.append({
            "age": int(row["age_years"]),
            "sex": "Male" if row["is_male"] == 1 else "Female",
            "height": int(row["height"]),
            "weight": round(float(row["weight"]), 1),
            "blood_pressure": f"{int(row['ap_hi'])}/{int(row['ap_lo'])}",
            "cholesterol": LEVELS.get(int(row["cholesterol"]), row["cholesterol"]),
            "smoke": "Yes" if row["smoke"] == 1 else "No",
            "active": "Yes" if row["active"] == 1 else "No",
            "has_disease": bool(knn_index["labels"].iloc[pos]),
            "distance": round(float(distance), 2),
        })

    n_disease = sum(n["has_disease"] for n in neighbours)
    return {
        "k": k,
        "n_disease": n_disease,
        "summary": f"{n_disease} out of your {k} most similar patients had heart disease",
        "neighbours": neighbours,
    }