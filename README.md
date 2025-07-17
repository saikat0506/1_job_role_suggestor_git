Here’s a clean and professional version of your `README.md` formatted with proper markdown syntax, emojis, and section headers:

---

# 🎯 AI Job Role Suggester

<div align="center">
  <strong>A Full-Stack Machine Learning Application for Resume Analysis and Job Role Prediction</strong>
</div>

---

## 🚀 Live Demo

* 🔗 **Frontend (Vercel):** [job-role-suggestor.vercel.app](https://job-role-suggestor.vercel.app/)
* 🔗 **Backend API Docs (Render):** [job-role-suggestor-api.onrender.com/docs](https://job-role-suggestor-api.onrender.com/docs)

---

## 📚 Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Tech Stack & Architecture](#tech-stack--architecture)
* [Machine Learning Workflow](#machine-learning-workflow)
* [Local Setup & Installation](#local-setup--installation)
* [Deployment](#deployment)

---

## 🧠 Project Overview

The **AI Job Role Suggester** bridges the gap between a candidate’s qualifications and career opportunities. Users can upload their resume (PDF), and the system performs a detailed analysis using:

1. **Google Gemini API** for skill extraction
2. **XGBoost ML model** for predicting the top 3 most suitable job roles with confidence scores

This project showcases a full end-to-end machine learning system with modular design and secure deployment.

---

## ✨ Features

* 📄 **PDF Resume Parsing**
  Extracts text from resumes client-side using `pdf.js`.

* 🤖 **AI-Powered Skill Extraction**
  Uses Google Gemini API (`gemini-1.5-flash-latest`) to detect technical, soft, and software skills.

* 🧠 **ML-Powered Job Prediction**
  Predicts job roles using a custom-trained XGBoost model across 80 professions.

* 📊 **Confidence Scoring**
  Shows model certainty using percentage scores.

* 🔒 **Secure, Decoupled Architecture**
  Frontend and backend are separated; sensitive keys are never exposed to users.

* 🧪 **Interactive API Docs**
  Auto-generated Swagger UI using FastAPI.

---

## 🧱 Tech Stack & Architecture

### 🎨 Frontend:

* **Framework:** Vanilla JavaScript + HTML5
* **Styling:** Tailwind CSS (via CDN)
* **Libraries:** pdf.js
* **Deployment:** Vercel (Static Hosting)

### ⚙️ Backend:

* **Framework:** FastAPI
* **Language:** Python 3.10+
* **Server:** Uvicorn with Gunicorn
* **Deployment:** Render

### 🧠 Machine Learning:

* **Libraries:** Scikit-learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **External API:** Google Gemini

---

## 🧪 Machine Learning Workflow

* **Data Generation:**
  A synthetic dataset with 50,000 samples across 80 job roles was generated using a weighted-skill methodology.

* **Model Training:**
  Trained using `XGBClassifier` with `GridSearchCV` and 10-fold cross-validation.

* **Serialization:**
  Final model, label encoder, and skill feature list saved using `joblib`.

---

## 🛠️ Local Setup & Installation

### 🧩 Prerequisites

* Python 3.10+
* An NVIDIA GPU with CUDA (for training, optional)
* A valid Google Gemini API Key

---

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd 1_JOB_ROLE_SUGGESTOR_GIT
```

---

### ⚙️ 2. Set Up the Backend

```bash
# Navigate to backend
cd JOB_Role_Suggestor_ML_backend

# Install requirements (shared with frontend folder)
pip install -r ../JOB_Role_Suggestor_frontend/requirements.txt

# Set your Google Gemini API key
# For Windows (CMD)
set GOOGLE_API_KEY="YOUR_SECRET_KEY_HERE"

# For macOS/Linux (bash)
export GOOGLE_API_KEY="YOUR_SECRET_KEY_HERE"

# Start FastAPI server
uvicorn API_XGboost:app --reload --port 8000
```

🔗 Visit your local API at: `http://127.0.0.1:8000`

---

### 🌐 3. Run the Frontend

* No server is required for the frontend.
* Just open `index.html` from the `JOB_Role_Suggestor_frontend` folder directly in your browser.

Make sure the `apiUrl` in `index.html` points to:

```
http://127.0.0.1:8000/api/process_resume
```

---

## ☁️ Deployment

### 🔧 Backend (Render)

* Deployed as a web service using **Render**
* Runs Gunicorn + Uvicorn worker within free-tier memory limits
* `GOOGLE_API_KEY` set securely in the Render dashboard

### 🎨 Frontend (Vercel)

* Deployed as a **static site**
* `apiUrl` in `index.html` points to the Render backend URL

---

## 💡 Final Note

This project was built as a full demonstration of deploying a modern, ML-powered, cloud-native application — combining **AI APIs, machine learning, and full-stack web dev** into one seamless user experience.

---

Let me know if you’d like me to generate badges (build, deploy, license, etc.) or add a demo GIF to this `README.md`!
