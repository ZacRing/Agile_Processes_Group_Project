from django.shortcuts import render

# Create your views here.
from pathlib import Path
import joblib
import pandas as pd
from django.shortcuts import render
from .forms import HeartDiseaseForm

# Load the saved tuned Logistic Regression model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "model1_logistic_regression_tuned.joblib"

model = joblib.load(MODEL_PATH)

def home(request):
    result = None
    if request.method == "POST":
        form = HeartDiseaseForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data

            # Convert form values into numbers
            input_data = {
                "age_years": data["age_years"],
                "is_male": int(data["is_male"]),
                "height": data["height"],
                "weight": data["weight"],
                "ap_hi": data["ap_hi"],
                "ap_lo": data["ap_lo"],
                "cholesterol": int(data["cholesterol"]),
                "gluc": int(data["gluc"]),
                "smoke": int(data["smoke"]),
                "alco": int(data["alco"]),
                "active": int(data["active"]),
            }

            # Create a DataFrame in the exact feature order
            input_df = pd.DataFrame(
                [input_data],
                columns=model.feature_names_in_
            )

            # Get prediction and probability for class 1
            prediction = model.predict(input_df)[0]
            class_1_index = list(model.classes_).index(1)
            probability = model.predict_proba(input_df)[0][class_1_index]

            result = {
                "prediction": (
                    "Higher predicted risk"
                    if prediction == 1
                    else "Lower predicted risk"
                ),
                "probability": round(probability * 100, 1),
            }
    else:
        form = HeartDiseaseForm()

    return render(
        request,
        "predictions/index.html",
        {"form": form, "result": result}
    )