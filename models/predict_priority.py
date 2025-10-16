# models/predict_priority.py

import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

def predict_task_priority(deadline, duration, is_flexible, category, model_path='models/task_priority_model.pkl'):
    """Predict task priority score based on input features."""

    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model not found. Please train it first.")

    # Load trained components
    model_data = joblib.load(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    encoder = model_data["encoder"]

    # Convert deadline to numeric feature (hours from now)
    deadline_hours = (datetime.strptime(deadline, "%Y-%m-%d %H:%M") - datetime.now()).total_seconds() / 3600

    # Encode category
    category_encoded = encoder.transform([category])[0] if category in encoder.classes_ else 0

    # Prepare input features
    X_new = np.array([[deadline_hours, duration, is_flexible, category_encoded]])
    X_scaled = scaler.transform(X_new)

    # Predict score
    predicted_score = model.predict(X_scaled)[0]
    predicted_score = max(0, min(100, predicted_score))  # Clamp between 0 and 100

    return round(predicted_score, 2)
