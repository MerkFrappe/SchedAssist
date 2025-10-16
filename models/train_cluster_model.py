# models/train_cluster_model.py

import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

def train_task_clusters(db_path='tasks.db', model_path='models/task_cluster_model.pkl'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM tasks", conn)
    conn.close()

    # Ensure score column exists
    if 'score' not in df.columns:
        raise ValueError("Database must have a 'score' column for training.")

    # Prepare data (you can add more features later)
    data = df[['score']]

    # Train KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(data)

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(kmeans, model_path)

    print("✅ Task clustering model trained and saved successfully at:", model_path)

    # Optional: show average score per cluster
    print(df.groupby('cluster')['score'].mean())

    return df
