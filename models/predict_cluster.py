# models/predict_cluster.py

import joblib
import numpy as np
import os

def predict_task_cluster(score, model_path='models/task_cluster_model.pkl'):
    """
    Predicts which cluster (0,1,2) the task belongs to based on score.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Trained model not found. Train it first using train_cluster_model.py")

    kmeans = joblib.load(model_path)
    cluster = kmeans.predict(np.array([[score]]))[0]

    return int(cluster)
