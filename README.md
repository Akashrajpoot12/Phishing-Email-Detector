# Phishing Email Detector

A machine learning-based web application that detects phishing emails. Users can paste an email's content into a web interface or call an API to check whether the email is **phishing** or **legitimate**.

---

## Features

- Train a **TF-IDF + Logistic Regression** model to classify emails.
- Clean and preprocess email text automatically.
- Web interface via **Flask**:
  - Paste email text and get a **prediction**.
  - View the **phishing probability** and cleaned text.
- REST API for programmatic access (`/api/predict`).
- Easily extendable with larger datasets and stronger models (XGBoost, BERT, etc.).
- Demo dataset included if `emails.csv` is missing.

---

## Folder Structure

phishing_email_detector/
│
├── emails.csv # Dataset (columns: text,label)
├── phish_detector.py # Main Python script (training + Flask app)
├── templates/
│ └── index.html # Optional: HTML template for Flask
└── requirements.txt # Python dependencies



---

## Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd phishing_email_detector





python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows



pip install -r requirements.txt
Dataset

emails.csv should have two columns:

text: email content

label: 1 = phishing, 0 = legitimate

text,label
"Dear user, your account is suspended. Click http://phish.example to reactivate",1
"Team meeting at 10 AM tomorrow. Please confirm attendance.",0
Running the Project
1. Run the Flask Web App
python phish_detector.py


Open your browser at http://localhost:5000

Paste an email text and click Check Email

See the prediction (PHISHING / LEGITIMATE) and phishing probability

2. API Usage

Send a POST request to the REST API:

curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"Your account is suspended. Click here http://phish.example"}'


Sample JSON response:

{
  "label": 1,
  "probability": 0.9834,
  "clean_text": "your account is suspended click here <URL>"
}
