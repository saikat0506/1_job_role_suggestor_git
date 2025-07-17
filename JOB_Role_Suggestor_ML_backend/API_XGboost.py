import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import json

# --- Pydantic Models for a clear API schema ---
class Suggestion(BaseModel):
    role: str
    confidence: str

class PredictionResponse(BaseModel):
    predicted_role: str
    suggestions: List[Suggestion]
    extracted_keywords: List[str]

class ResumeRequest(BaseModel):
    resume_text: str

# --- Initialize the FastAPI App ---
app = FastAPI(
    title="Secure Job Role Prediction API",
    description="Processes resume text to extract skills via Gemini and predicts job roles via XGBoost.",
    version="3.1.0" # Version updated
)

# --- Add CORS Middleware for frontend access ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variables for models and assets ---
model = None
feature_list = None
label_encoder = None
feature_lookup = {}
gemini_model = None

@app.on_event("startup")
def load_assets():
    """Load all models and configure APIs at server startup."""
    global model, feature_list, label_encoder, feature_lookup, gemini_model
    
    # 1. Configure Gemini API from environment variable
    try:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            print("CRITICAL WARNING: GOOGLE_API_KEY environment variable not found.")
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to configure Gemini API: {e}")

    # 2. Load XGBoost Model Assets
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(script_dir, 'model_training', 'saved_model_xgboost_gpu')
        
        print(f"Attempting to load XGBoost assets from: '{MODEL_DIR}'")
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"Directory not found: '{MODEL_DIR}'")

        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_job_predictor')], reverse=True)
        feature_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_feature_list')], reverse=True)
        encoder_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('xgb_label_encoder')], reverse=True)

        if not all([model_files, feature_files, encoder_files]):
            raise FileNotFoundError("One or more required XGBoost asset files were not found.")

        model = joblib.load(os.path.join(MODEL_DIR, model_files[0]))
        feature_list = joblib.load(os.path.join(MODEL_DIR, feature_files[0]))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, encoder_files[0]))
        feature_lookup = {f.lower(): f for f in feature_list}
        
        print("All XGBoost assets loaded successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: XGBoost model could not be loaded: {e}")
        model = None

# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "Secure Job Role Prediction API is running."}

# --- The All-in-One Secure Prediction Endpoint ---
@app.post("/api/process_resume", response_model=PredictionResponse)
async def process_resume_and_predict(request: ResumeRequest):
    if not all([model, feature_list, label_encoder, gemini_model]):
        raise HTTPException(status_code=503, detail="A required model or API client is not loaded. Check server logs.")

    # Step 1: Extract Keywords with Gemini (Securely on the Backend)
    try:
        prompt = f"""From the following resume text, extract a list of all relevant professional skills (technical skills, software, and soft skills). Return the skills as a simple JSON array of strings. For example: ["Python", "Project Management", "AWS"]. Resume text: "{request.resume_text}" """
        response = await gemini_model.generate_content_async(prompt)
        
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        keywords = json.loads(cleaned_text)
        print(f"Gemini extracted keywords: {keywords}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")

    # Step 2: Predict Job Role with XGBoost
    try:
        applicant_profile = {feature: 0.0 for feature in feature_list}
        for skill in keywords:
            lower_skill = skill.lower()
            if lower_skill in feature_lookup:
                applicant_profile[feature_lookup[lower_skill]] = 1.0
        
        applicant_df = pd.DataFrame([applicant_profile])[feature_list]
        
        probabilities = model.predict_proba(applicant_df)
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        
        predicted_role_label = label_encoder.inverse_transform([top_3_indices[0]])[0]
        
        suggestions = []
        for index in top_3_indices:
            role_label = label_encoder.inverse_transform([index])[0]
            confidence = probabilities[0][index]
            suggestions.append({"role": role_label, "confidence": f"{confidence:.2%}"})

        return {
            "predicted_role": predicted_role_label,
            "suggestions": suggestions,
            "extracted_keywords": keywords
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during local model prediction: {e}")

if __name__ == '__main__':
    uvicorn.run("API_XGboost:app", host="0.0.0.0", port=8000, reload=True)
