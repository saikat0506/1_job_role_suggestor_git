Technical Documentation: AI Job Role Suggester
Version: 1.0
Date: July 17, 2025

1. Introduction
1.1. Project Overview
The AI Job Role Suggester represents a full-stack web application engineered to perform analytical processing of user-submitted resumes for the purpose of predicting suitable professional roles. The system utilizes a custom-trained machine learning model for role prediction and the Google Gemini API for dynamic skill extraction, providing users with substantive career guidance. The core objective is to reconcile the disparity between a candidate's listed skills and the diverse landscape of available career opportunities, thereby enhancing the efficiency and scope of their job search.

1.2. Problem Statement
Within the contemporary labor market, prospective employees frequently encounter substantial obstacles in identifying the complete spectrum of occupational roles for which their qualifications are appropriate. This difficulty is compounded by the rapid evolution of job titles and the increasing importance of transferable skills across different industries. Resumes contain a wealth of information, but manually mapping the nuanced combination of technical and soft skills to the vast landscape of job titles is a time-consuming, subjective, and often incomplete process. Such a manual methodology may result in overlooked opportunities and an unduly restricted scope in the job search process. The objective of this application is to automate this discovery process, thereby providing objective, data-driven recommendations to guide and expand the user's career search.

1.3. Solution Overview
The solution is architected as a decoupled, client-server application to ensure scalability, maintainability, and a clear separation of concerns. A static frontend, constructed utilizing HTML and vanilla JavaScript to ensure maximal compatibility and performance, furnishes the user interface for document submission. The backend architecture consists of a high-performance, asynchronous Python Application Programming Interface (API) developed with the FastAPI framework. This API orchestrates the core logic, which includes securely calling the Google Gemini API for skill extraction from resume text and employing a custom-trained XGBoost model hosted on the server to infer job role probabilities from the extracted skills. This architectural design guarantees that all computationally demanding operations and sensitive data handling, such as the management of the secret API key, are encapsulated within the secure server environment, thereby providing a robust and secure user experience.

2. System Architecture
2.1. High-Level Architecture Diagram
The following diagram illustrates the end-to-end data flow, from user interaction to the final display of results.

[User's Browser]
      |
      | 1. A user selects a Portable Document Format (PDF) file. Client-side
      |    JavaScript, leveraging the pdf.js library, subsequently parses
      |    the document to extract its raw, unstructured textual content.
      v
[Frontend: Vercel (index.html)]
      |
      | 2. An asynchronous POST request, containing the extracted text within
      |    a JSON payload structured as { "resume_text": "..." }, is
      |    transmitted to the backend API.
      v
[Backend API: Render (API_XGboost.py)]
      |
      | 3. The backend API receives the textual data and initiates a secure
      |    call to the Google Gemini API, authenticating via a server-side
      |    environment variable that stores the confidential API key.
      v
[Google Gemini API]
      |
      | 4. The Gemini API processes the text against a predefined prompt,
      |    returning a structured JSON array composed of identified
      |    professional skills.
      v
[Backend API: Render]
      |
      | 5. The backend API then preprocesses the received keyword list,
      |    transforming it into a binary feature vector that conforms to the
      |    input specifications of the machine learning model.
      v
[ML Model (model.joblib)]
      |
      | 6. The XGBoost model computes a probability distribution across the 80
      |    potential job roles. Subsequently, a pre-fitted LabelEncoder
      |    object is utilized to decode the model's numeric output into
      |    their corresponding human-readable job titles.
      v
[Backend API: Render]
      |
      | 7. The API formats the three highest-probability predictions, along
      |    with the complete list of extracted keywords, into a final JSON
      |    object for transmission.
      v
[Frontend: Vercel]
      |
      | 8. The client-side JavaScript parses the received JSON response and
      |    dynamically manipulates the Document Object Model (DOM) to render
      |    the results within the user's browser.
      v
[User's Browser]

2.2. Component Breakdown
Frontend: The frontend consists of a single-page application (index.html), which is deployed on Vercel's global edge network to ensure low-latency content delivery to users worldwide. Its responsibilities encompass all user interactions, client-side PDF parsing through the pdf.js library, and asynchronous communication with the backend API, which is managed via the fetch API. The application is constructed using standard HTML5, with styling implemented through the Tailwind CSS framework (delivered via a Content Delivery Network for simplicity), and its client-side logic is powered by vanilla JavaScript to maintain a minimal performance footprint.

Backend: The backend is a Python-based API developed using the FastAPI framework and deployed on the Render cloud platform. It exposes a singular, secure endpoint (/api/process_resume) designed to manage the complete analysis workflow. The selection of FastAPI as the framework was predicated on its high performance, native support for asynchronous operations, and its integral feature of automatic request and response validation through Pydantic models. This architectural choice guarantees that all sensitive operations, including the management of the Google Gemini API key, are securely contained within the server-side environment.

Machine Learning Model: The core predictive component is a highly optimized gradient-boosted decision tree model, specifically an XGBClassifier, which has been trained to classify a given set of skills into one of 80 predefined occupational roles. The XGBoost algorithm was selected due to its demonstrated efficacy in handling sparse, high-dimensional data and its established reputation for delivering state-of-the-art accuracy in classification tasks. The trained model object, along with its corresponding feature list and label encoder, are serialized using the joblib library and stored as binary files, which facilitates efficient loading into memory upon server initialization.

External Services: The system utilizes the Google Gemini API, specifically the gemini-1.5-flash-latest model, for its advanced natural language processing (NLP) capabilities. This external service is tasked with the dynamic extraction of a structured list of professional skills from unstructured resume text, which constitutes the critical initial phase of the application's analysis pipeline.

3. Machine Learning Workflow
The efficacy of the application is contingent upon a robust, custom-trained machine learning model. The workflow is divided into two main phases: data generation and model training.

3.1. Data Generation
Objective: The primary objective of the data generation phase was to construct a large-scale, high-fidelity dataset for supervised learning. This dataset was engineered to establish a robust and unambiguous statistical correlation between defined skill sets and specific occupational roles, with the goal of minimizing prediction uncertainty in the final model.

Methodology (generate_dataset.ipynb):

Role & Skill Definition: A curated taxonomy of 80 distinct occupational roles was established, comprising a strategic allocation of 30 roles from the technology sector and 50 from other prominent global professions. For each role, a set of primary (core, defining skills), secondary (related but not essential skills), and soft skills was manually assigned to create a logical foundation.

Weighted Feature Generation: To create a dataset with greater nuance than a simple binary (0/1) feature representation, a skill-weighting methodology was implemented. This approach was identified as a critical step to mitigate the low-confidence predictions observed in preliminary model iterations. For each generated profile:

Primary skills were assigned a high weight (randomly between 0.8 and 1.0) to create a strong signal.

Secondary skills were assigned a medium weight (0.4 to 0.7) to introduce realistic variance.

Soft skills were assigned a low weight (0.1 to 0.3) to reflect their supportive, rather than defining, nature.
This weighting methodology ensures that the training data reflects the differential importance of various skills for a given role, thereby enabling the model to learn more complex relationships and, consequently, produce more confident and accurate predictions.

Output: The output of this phase is a CSV file (job_skills_80_roles_weighted.csv) containing 50,000 samples. In this dataset, each row corresponds to a synthetic professional profile, each feature column represents a specific skill's assigned weight, and the final column serves as the target variable, indicating the associated job role.

3.2. Model Training and Evaluation
Objective: The objective of the model training phase was to develop a highly accurate and confident classifier for the prediction task, with optimizations for both predictive performance and computational reliability.

Methodology (training_XGboost.ipynb):

Algorithm Selection: The XGBClassifier (XGBoost) algorithm was selected in favor of alternatives such as RandomForestClassifier due to its consistently superior performance on structured, tabular data. Its gradient boosting mechanism, which builds trees sequentially to correct the errors of previous trees, typically results in higher accuracy for complex, multi-class classification tasks like this one.

Data Preprocessing: A critical data preprocessing step involved the transformation of categorical Job_Role text labels (e.g., "Data Scientist") into their corresponding integer representations (e.g., 15) through the use of the sklearn.preprocessing.LabelEncoder. This encoding is a mandatory prerequisite for the XGBoost algorithm, which operates exclusively on numerical data.

Hyperparameter Tuning: To ascertain the optimal model configuration, the sklearn.model_selection.GridSearchCV utility was employed. This utility performs an exhaustive search over a predefined hyperparameter space, testing a wide grid of key hyperparameters (such as n_estimators, max_depth, learning_rate, subsample, and colsample_bytree) using 10-fold cross-validation (cv=10). This robust evaluation method ensures that the selected parameters generalize well to unseen data and are not overfitted to a specific random split of the training set.

GPU Acceleration: To mitigate the significant computational expense associated with this exhaustive search, the training process was configured to leverage an NVIDIA Graphics Processing Unit (GPU) by setting the device="cuda" parameter within the XGBClassifier. This configuration offloads the intensive matrix computations to the GPU's parallel processing architecture, thereby reducing training duration from hours to minutes.

Output: The culmination of the training process is the serialization and storage of three critical artifacts as .joblib files:

xgb_job_predictor_... .joblib: The fully trained and optimized XGBoost model object, ready for prediction.

xgb_feature_list_... .joblib: The ordered list of skill names (column headers) that the model was trained on. This is essential to ensure that prediction inputs have the exact same structure.

xgb_label_encoder_... .joblib: The LabelEncoder object itself. This is crucial for the API to convert the model's numeric output (e.g., 15) back into the human-readable job role name ("Data Scientist").

3.3. Predictable Job Roles
The machine learning model has been trained to recognize and predict the following 80 distinct professional roles, categorized for clarity.

Technology Roles (30)

AI Ethics Specialist

Backend Developer

Blockchain Developer

Cloud Engineer

Cybersecurity Analyst

Data Analyst

Data Engineer

Data Scientist

Database Administrator (DBA)

DevOps Engineer

Firmware Engineer

Frontend Developer

Full-Stack Developer

Game Developer

IT Support Specialist

Machine Learning Engineer

Mobile App Developer

Network Engineer

Penetration Tester

Product Manager (Tech)

QA Engineer

Quantum Computing Scientist

Robotics Engineer

Salesforce Developer

Security Engineer

Site Reliability Engineer (SRE)

Software Engineer

Solutions Architect

Systems Administrator

UX/UI Designer

Other Top Professions (50)

Accountant

Actuary

Aerospace Engineer

Architect

Artist

Biologist

Biomedical Engineer

Business Analyst

Carpenter

Chef

Chemical Engineer

Chemist

Civil Engineer

Compliance Officer

Construction Manager

Content Strategist

Dentist

Digital Marketer

Economist

Electrical Engineer

Electrician

Environmental Engineer

Event Planner

Executive Assistant

Fashion Designer

Financial Analyst

Firefighter

Fitness Trainer

Geologist

Graphic Designer

HR Manager

Industrial Designer

Interior Designer

Investment Banker

Journalist

Lawyer

Librarian

Management Consultant

Marketing Manager

Mechanical Engineer

Medical Lab Scientist

Musician

Operations Manager

Paralegal

Pharmacist

Photographer

Physical Therapist

Physician

Physicist

Pilot

Plumber

Police Officer

Professor

Project Manager (Non-Tech)

Psychologist

Public Health Official

Public Relations Specialist

QA Engineer

Radiologic Technologist

Real Estate Agent

Recruiter

Registered Nurse

Research Scientist

Sales Manager

Social Worker

Statistician

Supply Chain Manager

Teacher

Technical Writer

Translator

Urban Planner

Veterinarian

Welder

4. Backend API (FastAPI)
The backend is the core of the application, handling all business logic and secure communications.

Implementation File: API_XGboost.py

Framework: The FastAPI framework was selected for its high-throughput performance, native support for asynchronous I/O operations, and its automatic generation of interactive API documentation compliant with the OpenAPI standard (rendered via Swagger UI). Furthermore, its integration with the Pydantic library enables rigorous, type-hint-based data validation, which ensures that all incoming requests and outgoing responses adhere to a strictly defined data schema.

Startup Event Logic (@app.on_event("startup")):

To optimize for response latency and system readiness, all requisite assets are loaded into memory a single time upon server initialization, rather than on a per-request basis.

The GOOGLE_API_KEY is securely retrieved from the server's environment variables via the os.getenv() function. This is a critical security measure that prevents the exposure of secret credentials within the source code.

The three .joblib files (model, feature list, and label encoder) are loaded from the saved_model_xgboost directory. The application is designed to fail on startup if these critical assets cannot be found.

API Endpoints:

GET /: This endpoint serves as a basic health check, returning a static JSON response to confirm that the API service is operational.

POST /api/process_resume: This endpoint constitutes the primary operational interface of the API.

Request: The endpoint is designed to accept a JSON object containing a single key, resume_text, whose value is the full, unstructured text extracted from a user's PDF. The structure of this request body is strictly enforced by a Pydantic model.

Workflow:

The endpoint receives the resume text from the incoming request.

It initiates a secure, asynchronous call to the Google Gemini API, authenticating with the server-side key and transmitting the text within a formatted prompt engineered to elicit a JSON array response.

It parses the JSON array of professional keywords from the Gemini API's response.

It transforms the extracted keywords into a binary feature vector. This vector's dimensions and ordering align precisely with the feature set used during model training, and a case-insensitive lookup mechanism is employed to enhance matching robustness.

It invokes the model.predict_proba() method on the prepared feature vector to obtain a probability distribution across all 80 potential job roles.

The loaded label_encoder object is then utilized to translate the numeric indices of the highest-probability predictions back into their corresponding string-based job role names.

Finally, it formats the top three predictions into a PredictionResponse Pydantic object, which is then serialized into a JSON response and sent back to the client.

Error Handling: Error handling is managed through fastapi.HTTPException. This allows the API to return standardized, informative JSON error responses with appropriate HTTP status codes (e.g., 503 Service Unavailable if a required model is not loaded, or 500 Internal Server Error for other runtime failures).

5. Frontend (Single-Page HTML/JS)
The frontend is designed for simplicity, performance, and ease of deployment, providing a seamless user experience.

Implementation File: index.html

Technology Stack:

HTML5: Provides the foundational semantic structure for the web application.

Tailwind CSS (via CDN): All styling is managed through Tailwind CSS, delivered via a CDN. This utility-first framework facilitates the rapid development of a modern, responsive user interface by applying classes directly within the HTML markup, thereby obviating the need for external stylesheets.

Vanilla JavaScript: All client-side logic, including event handling, API communication, and DOM manipulation, is implemented using vanilla JavaScript. This approach was chosen to avoid the dependency overhead associated with larger frontend frameworks, thus ensuring minimal load times and optimal performance.

Key Libraries (via CDN):

pdf.js: The pdf.js library, an open-source project by Mozilla, is a critical component. It empowers the browser to directly open and parse the textual content of a user-submitted PDF document on the client side. This client-side processing strategy is pivotal as it eliminates the necessity of uploading the entire file to the server, which concurrently enhances application performance by reducing network payload and improves user privacy.

Application Flow (JavaScript):

An addEventListener attached to the file input element initiates the main operational workflow upon a user's file selection. The entire workflow is managed using async/await syntax to handle asynchronous operations, such as file reading and API calls, in a non-blocking manner, which prevents the user interface from becoming unresponsive.

The parsePdf() function leverages the browser's FileReader API in conjunction with the pdf.js library to asynchronously read the selected PDF file and extract its complete textual content.

The processResumeOnBackend() function executes a single fetch POST request to the secure /api/process_resume endpoint hosted on the Render backend. It transmits the extracted text within the request body and awaits the corresponding JSON response.

Upon the successful reception of a JSON response from the backend, the displayResults() and displaySkills() functions are invoked. These functions are responsible for the dynamic creation and insertion of HTML elements into the DOM, which serve to render the top prediction, alternative suggestions, and the list of extracted keywords in a structured and user-friendly layout. A try...catch block encapsulates the primary workflow to gracefully handle any potential runtime errors and display informative messages to the user.

6. Deployment
A decoupled, two-platform deployment strategy was implemented to optimize for scalability, security, and operational flexibility.

Backend (Render):

Platform Choice: The Render platform was selected for the backend deployment due to its developer-friendly interface, seamless integration with GitHub for continuous deployment, and its native support for Python web services.

Configuration: The service configuration specifies a Root Directory of JOB_Role_Suggestor_ML_backend to align with the project's repository structure. The Build Command is configured as pip install -r ../JOB_Role_Suggestor_frontend/requirements.txt to correctly resolve the path to the dependencies file, which resides outside the designated root directory.

Start Command: gunicorn serves as the production-ready Web Server Gateway Interface (WSGI) server. The -w 1 flag, which specifies a single worker process, is a critical configuration parameter necessary to operate within the memory constraints of Render's complimentary service tier. The full command is gunicorn -w 1 -k uvicorn.workers.UvicornWorker API_XGboost:app.

Environment Variables: The GOOGLE_API_KEY is managed as a secret environment variable within the Render dashboard. This adheres to industry-standard security practices for managing credentials in a production environment, ensuring that sensitive keys are never hardcoded into the application's source code.

Frontend (Vercel):

Platform Choice: The Vercel platform was selected for frontend deployment, primarily for its high-performance infrastructure for hosting static assets. Vercel's global Content Delivery Network (CDN) guarantees that users worldwide can access the index.html application with minimal latency.

Configuration: The Vercel project configuration specifies JOB_Role_Suggestor_frontend as the Root Directory, ensuring that deployment operations are correctly scoped to the frontend source files.

API URL: The apiUrl constant within the index.html JavaScript is hardcoded to the public URL of the deployed Render service (e.g., <https://job-role-suggestor-api.onrender.com/api/process_resume>). This variable constitutes the critical link that facilitates communication between the client-side application and the backend API.

Deployment Method: The deployment process is streamlined via Vercel's user interface, which supports a drag-and-drop mechanism for the frontend source folder, triggering an automatic build and deployment sequence.
