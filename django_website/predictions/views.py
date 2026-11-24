# Create your views here.
from pathlib import Path
import joblib
import pandas as pd
from django.shortcuts import render
from .forms import HeartDiseaseForm
from .ai_explaner_agent import explain_result
from django.http import JsonResponse

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
            # sort the percentage into low medium or high so its easy to understand
            # risk colour is the bootstrap colour for the badge green yellow or red
            percent = result["probability"]
            if percent < 35:
                result["risk_level"] = "Low"
                result["risk_colour"] = "success"
                result["message"] = (
                    "Your answers suggest a lower risk of heart disease. "
                    "Keep up healthy habits like staying active and eating well."
                )
            elif percent < 65:
                result["risk_level"] = "Medium"
                result["risk_colour"] = "warning"
                result["message"] = (
                    "Your answers suggest a moderate risk of heart disease. "
                    "It may be worth talking to a doctor about your blood pressure, "
                    "cholesterol and lifestyle."
                )
            else:
                result["risk_level"] = "High"
                result["risk_colour"] = "danger"
                result["message"] = (
                    "Your answers suggest a higher risk of heart disease. "
                    "We recommend speaking to a doctor for a proper check up."
                )

            # ask crewai and gemini to explain the result
            # save the answers and result so the ai can use them after the results page loads
            # float makes sure the number can be saved in the session
            request.session["input_data"] = input_data
            request.session["result"] = {
                "probability": float(result["probability"]),
                "risk_level": result["risk_level"],
            }

            # show the results page straight away the ai explanation loads in after
            return render(
                request,
                "predictions/results.html",
                {"result": result}
            )
    else:
        form = HeartDiseaseForm()

    return render(
        request,
        "predictions/index.html",
        {"form": form, "result": result}
    )

# the results page calls this after it loads to get the ai explanation
def ai_explanation(request):
    input_data = request.session.get("input_data")
    result = request.session.get("result")

    # if someone opens this without filling in the form theres nothing to explain
    if input_data is None or result is None:
        return JsonResponse({"explanation": None})

    # ask crewai and gemini to explain the result
    explanation = explain_result(input_data, result)
    return JsonResponse({"explanation": explanation})