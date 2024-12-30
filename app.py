from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import json

app = FastAPI()

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

lr_model = load_model("lr_model.pkl")
svm_model = load_model("svm_model.pkl")
rf_model = load_model("rf_model.pkl")
kmeans_model = load_model("kmeans_model.pkl")

# Input data model
class InputData(BaseModel):
    text: str

# API route for Logistic Regression Prediction
@app.post("/predict_lr/")
def predict_lr(data: InputData):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.transform([data.text])
        
        prediction = lr_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for SVM Prediction
@app.post("/predict_svm/")
def predict_svm(data: InputData):
    try:

        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.transform([data.text])
        
        prediction = svm_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for Random Forest Prediction
@app.post("/predict_rf/")
def predict_rf(data: InputData):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.transform([data.text])
        
        prediction = rf_model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API route for KMeans Clustering
@app.post("/predict_kmeans/")
def predict_kmeans(data: InputData):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.transform([data.text])
        
        cluster = kmeans_model.predict(features)
        return {"cluster": int(cluster[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))