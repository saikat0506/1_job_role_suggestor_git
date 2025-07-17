AI Job Role Suggester
<!-- Optional: You can create a banner image for your project -->

A full-stack web application that leverages a custom-trained machine learning model and the Google Gemini API to analyze resumes and suggest suitable job roles based on the user's skills.

Live Demo
Frontend (Vercel): <https://job-role-suggestor.vercel.app/>

Backend API Docs (Render): <https://job-role-suggestor-api.onrender.com/docs>

Table of Contents
Project Overview

Features

Tech Stack & Architecture

Machine Learning Workflow

Local Setup & Installation

Deployment

Project Overview
The AI Job Role Suggester is designed to bridge the gap between a candidate's qualifications and the vast landscape of career opportunities. A user can upload their resume in PDF format, and the application will perform a comprehensive analysis. It first uses the Google Gemini API to extract a list of professional skills from the resume text. This list is then fed into a custom-trained XGBoost model, which predicts the top 3 most suitable job roles along with a confidence score for each.

This project demonstrates an end-to-end machine learning application, from data generation and model training to building a secure, decoupled web architecture and deploying it to the cloud.

Features
PDF Resume Parsing: Client-side parsing of PDF files using pdf.js to extract text content directly in the browser.

AI-Powered Skill Extraction: Utilizes the Google Gemini API (gemini-1.5-flash-latest) to dynamically identify technical, soft, and software skills from unstructured resume text.

ML-Powered Job Prediction: A custom-trained XGBoost model predicts the most relevant job roles from a corpus of 80 distinct professions.

Confidence Scoring: Provides a percentage-based confidence score for each prediction, helping users understand the model's certainty.

Decoupled & Secure Architecture: A secure frontend and backend separation ensures that all API keys and sensitive operations are handled on the server, never exposing them to the client.

Interactive API: The backend is fully documented with an interactive Swagger UI, generated automatically by FastAPI.

Tech Stack & Architecture
The application is built with a modern, decoupled architecture.

Frontend:

Framework: Vanilla JavaScript, HTML5

Styling: Tailwind CSS (via CDN)

Libraries: pdf.js

Deployment: Vercel

Backend:

Framework: FastAPI

Language: Python 3.10+

Server: Uvicorn with Gunicorn for production

Deployment: Render

Machine Learning:

Core Library: Scikit-learn, XGBoost

Data Handling: Pandas, NumPy

External API: Google Gemini

Machine Learning Workflow
Data Generation: A synthetic dataset of 50,000 samples across 80 job roles was created. A weighted-skill methodology was used to provide a nuanced representation of skill importance for each role.

Model Training: An XGBClassifier was trained on the weighted dataset. GridSearchCV with 10-fold cross-validation was used to find the optimal hyperparameters.

Model Serialization: The final trained model, feature list, and label encoder were serialized using joblib for use in the API.

Local Setup & Installation
To run this project on your local machine, follow these steps.

Prerequisites:

Python 3.10 or higher

An NVIDIA GPU with CUDA installed (for GPU-accelerated training)

A Google Gemini API Key

1. Clone the Repository

git clone <https://github.com/your-username/your-repo-name.git>
cd 1_JOB_ROLE_SUGGESTOR_GIT

2. Set up the Backend

# Navigate to the backend directory

cd JOB_Role_Suggestor_ML_backend

# Install all required Python packages

# Note: The requirements.txt file is in the frontend folder

pip install -r ../JOB_Role_Suggestor_frontend/requirements.txt

# Set your Gemini API Key as an environment variable

# On Windows (Command Prompt)

set GOOGLE_API_KEY="YOUR_SECRET_KEY_HERE"

# On macOS/Linux

export GOOGLE_API_KEY="YOUR_SECRET_KEY_HERE"

# Run the FastAPI server

# This uses the secure version of the app

uvicorn API_XGboost:app --reload --port 8000

Your backend API should now be running at <http://127.0.0.1:8000>.

3. Run the Frontend
The frontend is a simple index.html file and does not require a separate server.

Make sure you are using the secure version of index.html that points to <http://127.0.0.1:8000/api/process_resume>.

Simply open the index.html file (located in the JOB_Role_Suggestor_frontend folder) directly in your web browser.

Deployment
The application is deployed on two separate platforms for optimal performance and security:

The FastAPI backend is deployed as a Web Service on Render. It is configured to use a single Gunicorn worker to stay within the free tier's memory limits. The GOOGLE_API_KEY is set as a secret environment variable in the Render dashboard.

The HTML/JS frontend is deployed as a static site on Vercel. The apiUrl variable in the index.html file is updated to point to the live Render backend URL.

This project was developed as a comprehensive demonstration of a full-stack machine learning application.
