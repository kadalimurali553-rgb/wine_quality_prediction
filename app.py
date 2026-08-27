
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

DEFAULTS = {
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11,
    "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4
}

model = None
metrics = {}
classes = []

def train_model():
    global model, metrics, classes
    csv_path = os.path.join("data", "winequality-red.csv")
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path, sep=";")
    missing = [c for c in FEATURES + ["quality"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df.dropna().drop_duplicates()
    X = df[FEATURES]
    y = df["quality"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=2.0, gamma="scale"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
        "samples": int(len(df)),
        "features": len(FEATURES)
    }
    classes = sorted(int(x) for x in model.named_steps["svm"].classes_)
    return True

try:
    train_model()
except Exception as exc:
    print("Model training error:", exc)

@app.route("/")
def index():
    return render_template("index.html", defaults=DEFAULTS, metrics=metrics, ready=model is not None)

@app.route("/api/status")
def status():
    return jsonify({"ready": model is not None, "metrics": metrics, "classes": classes})

@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "SVM model is not ready. Add data/winequality-red.csv and restart the app."
        }), 503

    payload = request.get_json(silent=True) or {}
    try:
        values = [float(payload[name]) for name in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide all 11 physicochemical features as numbers."}), 400

    row = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(row)[0])

    if prediction >= 7:
        label = "Excellent"
    elif prediction >= 6:
        label = "Good"
    elif prediction >= 5:
        label = "Average"
    else:
        label = "Needs Improvement"

    return jsonify({
        "quality": prediction,
        "label": label,
        "features": dict(zip(FEATURES, values))
    })

if __name__ == "__main__":
    app.run(debug=True)
