import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.cleaning import load_data, clean_data
from src.model import prepare_data

app = FastAPI(title="API Prédiction Diabète")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = load_data('data/diabetes.csv')
df = clean_data(df)
_, _, _, _, scaler = prepare_data(df)
model = torch.load('models/model.pt', weights_only=False)
model.eval()

class Patient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "API Prédiction Diabète - Bienvenue !"}

@app.get("/health")
def health():
    return {"status": "API en ligne"}

@app.post("/predict")
def predict(patient: Patient):
    data = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])
    data = scaler.transform(data)
    tensor = torch.FloatTensor(data)
    with torch.no_grad():
        output = model(tensor).item()
    prediction = 1 if output >= 0.5 else 0
    label = "Diabétique" if prediction == 1 else "Non diabétique"
    return {
        "prediction": prediction,
        "label": label,
        "probabilité": round(output * 100, 2)
    }