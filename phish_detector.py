import os
import re
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------
# Paths
DATA_PATH = "emails.csv"   
MODEL_PATH = "phish_model.joblib"

# -------------------------
# Text cleaning function
def clean_text(s):
    if pd.isna(s): return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " <URL> ", s)
    s = re.sub(r"\S+@\S+", " <EMAIL> ", s)
    s = re.sub(r"\d+", " <NUM> ", s)
    s = re.sub(r"[^a-z0-9<> ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# Load dataset
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise SystemExit("CSV must have 'text' and 'label' columns.")
else:
    df = pd.DataFrame({
        "text": [
            "Dear user, your account has been suspended. Click here http://phish.example to reactivate",
            "Meeting tomorrow at 10am — please confirm attendance",
            "Your invoice is attached. Please pay immediately: http://malicious.example/pay",
            "Lunch at 1? Let's meet in the cafeteria."
        ],
        "label": [1, 0, 1, 0]
    })
    print("No dataset found; using demo dataset.")

df['clean_text'] = df['text'].apply(clean_text)

# -------------------------
# Train-test split
X = df['clean_text'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------
# TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else None
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))
if y_proba is not None and len(np.unique(y_test)) > 1:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# -------------------------
# Flask app
app = Flask(__name__)
MODEL = pipeline

HTML = """
<!doctype html>
<title>Phishing Email Detector</title>
<h2>Phishing Email Detector</h2>
<form method=post>
  <textarea name=email_text rows=12 cols=80 placeholder="Paste full email text here..."></textarea><br>
  <input type=submit value="Check Email">
</form>
{% if result is defined %}
  <h3>Result: {{ result }}</h3>
  <p>Probability phishing: {{ prob }}</p>
  <pre>{{ cleaned }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    prob = None
    cleaned = None
    if request.method == "POST":
        txt = request.form.get("email_text","")
        cleaned = clean_text(txt)
        p = MODEL.predict_proba([cleaned])[0][1] if hasattr(MODEL, "predict_proba") else MODEL.predict([cleaned])[0]
        label = MODEL.predict([cleaned])[0]
        result = "PHISHING" if int(label) == 1 else "LEGITIMATE"
        prob = float(p) if isinstance(p, (float, np.floating)) else float(p[1]) if hasattr(p, "__len__") else float(p)
        prob = f"{prob:.4f}"
    return render_template_string(HTML, result=result, prob=prob, cleaned=cleaned)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    text = data.get("text","")
    ct = clean_text(text)
    label = int(MODEL.predict([ct])[0])
    prob = float(MODEL.predict_proba([ct])[0][1]) if hasattr(MODEL, "predict_proba") else None
    return jsonify({"label": label, "probability": prob, "clean_text": ct})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
