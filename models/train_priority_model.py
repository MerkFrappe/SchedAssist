# models/train_priority_model.py

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def train_priority_model(db_path='tasks.db', model_path='models/task_priority_model.pkl'):
    """Train a predictive model to estimate task priority scores."""

    # Connect to database
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM tasks", conn)
    conn.close()

    # Drop rows with missing values in important columns
    df = df.dropna(subset=['deadline', 'category', 'duration', 'is_flexible', 'score'])

    # -----------------------------
    # 🔹 FEATURE ENGINEERING
    # -----------------------------
    # Convert deadline to number of hours from now
    df['deadline_hours'] = pd.to_datetime(df['deadline']).map(lambda x: (x - pd.Timestamp.now()).total_seconds() / 3600)

    # Encode category (text → numbers)
    encoder = LabelEncoder()
    df['category_encoded'] = encoder.fit_transform(df['category'])

    # Select features (X) and label (y)
    X = df[['deadline_hours', 'duration', 'is_flexible', 'category_encoded']]
    y = df['score']

    # Standardize data for better model stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model and preprocessing objects
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "encoder": encoder
    }, model_path)

    print("✅ Priority prediction model trained and saved successfully!")
    print(f"R² Score on test set: {model.score(X_test, y_test):.3f}")

    return model
