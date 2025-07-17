import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os

# --- Pydantic Models for API Schema ---
class Suggestion(BaseModel):
    role: str
    confidence: str

class PredictionResponse(BaseModel):
    predicted_role: str
    suggestions: List[Suggestion]

class SkillRequest(BaseModel):
    skills: List[str]

# --- Initialize the FastAPI App ---
app = FastAPI(
    title="XGBoost Job Role Prediction API",
    description="A high-performance API using an XGBoost model to predict job roles from skills.",
    version="2.0.0"
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variables for the model components ---
model = None
feature_list = None
label_encoder = None
feature_lookup = {}

@app.on_event("startup")
def load_model_assets():
    """
    Load the XGBoost model, feature list, and label encoder at application startup.
    This function runs once when the server starts.
    """
    global model, feature_list, label_encoder, feature_lookup
    try:
        # Construct the absolute path to the XGBoost model directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(script_dir, 'model_training' ,'saved_model_xgboost')
        
        print(f"Attempting to load assets from directory: '{MODEL_DIR}'")
        
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"The directory '{MODEL_DIR}' was not found.")

        # Find the latest model, feature, and encoder files
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_job_predictor')], reverse=True)
        feature_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_feature_list')], reverse=True)
        encoder_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_label_encoder')], reverse=True)

        if not all([model_files, feature_files, encoder_files]):
            raise FileNotFoundError("One or more required model asset files (.joblib) were not found.")

        MODEL_PATH = os.path.join(MODEL_DIR, model_files[0])
        FEATURES_PATH = os.path.join(MODEL_DIR, feature_files[0])
        ENCODER_PATH = os.path.join(MODEL_DIR, encoder_files[0])

        print(f"Loading model: {MODEL_PATH}")
        print(f"Loading features: {FEATURES_PATH}")
        print(f"Loading encoder: {ENCODER_PATH}")

        model = joblib.load(MODEL_PATH)
        feature_list = joblib.load(FEATURES_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        
        # Create the case-insensitive lookup dictionary
        feature_lookup = {f.lower(): f for f in feature_list}
        
        print("All XGBoost assets loaded successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: Model assets could not be loaded at startup: {e}")
        # Clear globals on failure
        model, feature_list, label_encoder = None, None, None


# --- Define the Prediction API Endpoint ---
@app.post("/api/predict", response_model=PredictionResponse)
def predict_job_role(request: SkillRequest):
    """
    Predicts a job role based on a list of skills using the XGBoost model.
    """
    if not all([model, feature_list, label_encoder]):
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Model is not loaded. Please check server logs for critical errors."
        )

    user_skills = request.skills
    
    # Preprocess the input using the weighted approach (1 for present, 0 for absent)
    # The model was trained on weights, but for prediction, binary input is standard
    # unless the user provides their own weights.
    applicant_profile = {feature: 0.0 for feature in feature_list}
    matched_skills = []
    for skill in user_skills:
        lower_skill = skill.lower()
        if lower_skill in feature_lookup:
            original_cased_skill = feature_lookup[lower_skill]
            applicant_profile[original_cased_skill] = 1.0 # Use 1.0 to represent a present skill
            matched_skills.append(original_cased_skill)
            
    print(f"Received skills: {user_skills}")
    print(f"Matched skills for prediction: {matched_skills}")

    applicant_df = pd.DataFrame([applicant_profile])[feature_list]

    # Make a prediction
    try:
        # Predict probabilities
        probabilities = model.predict_proba(applicant_df)
        
        # Get the top 3 predictions
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        
        # Convert numeric predictions back to job role names using the encoder
        predicted_role_encoded = top_3_indices[0]
        predicted_role_label = label_encoder.inverse_transform([predicted_role_encoded])[0]

        suggestions = []
        for index in top_3_indices:
            role_label = label_encoder.inverse_transform([index])[0]
            confidence = probabilities[0][index]
            suggestions.append({
                "role": role_label,
                "confidence": f"{confidence:.2%}"
            })

        return {
            "predicted_role": predicted_role_label,
            "suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    return {"status": "XGBoost Job Role Prediction API is running."}


# --- Run the FastAPI App with Uvicorn ---
if __name__ == '__main__':
    uvicorn.run("app_xgboost:app", host="0.0.0.0", port=5000, reload=True)
