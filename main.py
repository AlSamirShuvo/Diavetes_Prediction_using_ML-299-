from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from ml_model import diabetes_model
import pandas as pd
import shap
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Or put ["http://127.0.0.1:5500"] if serving HTML via Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    gender: float
    age: float
    hypertension: float
    heart_disease: float
    smoking_history: float
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running."}

@app.post("/predict")
def predict(data: PatientData):
    user_data = [
        data.gender, data.age, data.hypertension, data.heart_disease,
        data.smoking_history, data.bmi, data.HbA1c_level, data.blood_glucose_level
    ]
    prediction = diabetes_model.predict(user_data)
    return {"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"}

advice_map = {
    'gender': None,  # skipped

    'age': (
        "As you get older, your diabetes risk increases. "
        "Stay active, eat balanced meals, and schedule regular checkups."
    ),

    'hypertension': (
        "High blood pressure can raise your diabetes and heart risks. "
        "You should cut down on salt, manage stress, and follow your doctor’s advice."
    ),

    'heart_disease': (
        "If you have heart problems, your diabetes risk is higher. "
        "Protect yourself with a heart-healthy diet, regular exercise, and routine checkups."
    ),

    'smoking_history': (
        "Smoking makes it harder for your body to use insulin. "
        "If you smoke, you should work on quitting; if you don’t, keep staying smoke-free."
    ),

    'bmi': (
        "Being overweight raises your diabetes risk. "
        "Even a small weight loss can help you—focus on healthy meals and daily activity."
    ),

    'HbA1c_level': (
        "Your HbA1c shows how your blood sugar has been over time. "
        "You can improve it with balanced meals, exercise, and regular monitoring."
    ),

    'blood_glucose_level': (
        "Your blood sugar level may be high from diet or stress. "
        "You can keep it steady by limiting sugary foods, eating more fiber, and staying active after meals."
    ),
}


@app.post("/explain")
def explain(data: PatientData):
    user_data = [
        data.gender, data.age, data.hypertension, data.heart_disease,
        data.smoking_history, data.bmi, data.HbA1c_level, data.blood_glucose_level
    ]
    df = pd.DataFrame([user_data], columns=diabetes_model.features)
    explainer = shap.TreeExplainer(diabetes_model.model)
    shap_values = explainer.shap_values(df)[0]  # SHAP values for input

    # Get top 3 positive contributors, exclude gender
    features_shap = list(zip(diabetes_model.features, shap_values))
    features_shap = [f for f in features_shap if f[0] != 'gender' and f[1] > 0]
    top_features = sorted(features_shap, key=lambda x: x[1], reverse=True)[:4]

    advice_list = []
    for feature, val in top_features:
        advice = advice_map.get(feature, "No advice available.")
        advice_list.append({
            "feature": feature,
            "contribution": float(val),
            "advice": advice
        })

    return {
        "top_risk_factors": advice_list
    }