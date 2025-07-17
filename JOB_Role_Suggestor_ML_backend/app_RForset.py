import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import os

# --- Pydantic Models for Clearer API Docs ---

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
    title="Job Role Prediction API",
    description="An API that predicts job roles based on a list of skills using a trained machine learning model.",
    version="1.2.0"
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load the Trained Model and Feature List ---
model = None
feature_list = None
feature_lookup = {}

@app.on_event("startup")
def load_model():
    """
    Load the model and feature list at application startup.
    This is a more robust way to handle model loading in FastAPI.
    """
    global model, feature_list, feature_lookup
    try:
        # Create a robust, absolute path to the model directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # ** THE FIX **: Changed 'model training' to 'model_training'
        MODEL_DIR = os.path.join(script_dir, 'model_training', 'saved_model')
        
        print(f"Attempting to load model from absolute directory: '{MODEL_DIR}'")
        
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(f"The directory '{MODEL_DIR}' was not found or is not a directory.")

        model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('job_role_predictor') and f.endswith('.joblib')]
        feature_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('feature_list') and f.endswith('.joblib')]

        if not model_files or not feature_files:
            raise FileNotFoundError("No model or feature files found in the specified directory.")

        latest_model_file = sorted(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)[0]
        latest_feature_file = sorted(feature_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)), reverse=True)[0]
        
        MODEL_PATH = os.path.join(MODEL_DIR, latest_model_file)
        FEATURES_PATH = os.path.join(MODEL_DIR, latest_feature_file)

        print(f"Loading model: {MODEL_PATH}")
        print(f"Loading features: {FEATURES_PATH}")

        model = joblib.load(MODEL_PATH)
        feature_list = joblib.load(FEATURES_PATH)
        
        feature_lookup = {f.lower(): f for f in feature_list}
        
        print("Model and feature list loaded successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: Model could not be loaded at startup: {e}")
        # In a real application, you might want the app to fail completely if the model doesn't load.
        model = None
        feature_list = None


# --- Define the Prediction API Endpoint ---
@app.post("/api/predict", response_model=PredictionResponse)
def predict_job_role(request: SkillRequest):
    """
    Predicts a job role based on a list of skills provided by the user.
    """
    if model is None or not feature_list:
        # ** THE FIX **: Use HTTPException for clear, standard error responses.
        raise HTTPException(
            status_code=503, # 503 Service Unavailable
            detail="Model is not loaded. Please check server logs for errors."
        )

    user_skills = request.skills
    
    # Preprocess the input
    applicant_profile = {feature: 0 for feature in feature_list}
    matched_skills = []
    for skill in user_skills:
        lower_skill = skill.lower()
        if lower_skill in feature_lookup:
            original_cased_skill = feature_lookup[lower_skill]
            applicant_profile[original_cased_skill] = 1
            matched_skills.append(original_cased_skill)
            
    print(f"Received skills: {user_skills}")
    print(f"Matched skills: {matched_skills}")

    applicant_df = pd.DataFrame([applicant_profile])[feature_list]

    # Make a prediction
    try:
        prediction = model.predict(applicant_df)
        probabilities = model.predict_proba(applicant_df)
        
        top_3_indices = probabilities[0].argsort()[-3:][::-1]
        top_3_roles = model.classes_[top_3_indices]
        top_3_probs = probabilities[0][top_3_indices]

        suggestions = []
        for role, prob in zip(top_3_roles, top_3_probs):
            suggestions.append({
                "role": role,
                "confidence": f"{prob:.2%}"
            })

        return {
            "predicted_role": prediction[0],
            "suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")


# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    return {"status": "Job Role Prediction API is running."}


# --- Run the FastAPI App with Uvicorn ---
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
