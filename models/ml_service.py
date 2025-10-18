# --- START OF FILE ml_service.py ---

# ml_service.py
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from datetime import datetime

class MLService:
    def __init__(self, db_path='tasks.db'):
        self.db_path = db_path
        self.cluster_model_path = 'models/task_cluster_model.pkl'
        self.priority_model_path = 'models/task_priority_model.pkl'
    
    def train_cluster_model(self):
        """Train task clustering model to find completion patterns"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()

        if df.empty or len(df) < 3: # Need at least 3 samples for 3 clusters
            print("⚠️ Not enough data to train cluster model.")
            return None

        # Ensure 'category' column exists before encoding
        if 'category' not in df.columns:
            print("⚠️ 'category' column not found in tasks table. Cannot train cluster model.")
            return None
            
        # Encode category for clustering
        encoder = LabelEncoder()
        df['category_encoded'] = encoder.fit_transform(df['category'])
        
        # --- FIX STARTS HERE ---
        # Define 'data' DataFrame for KMeans
        data = df[['duration', 'category_encoded', 'isCompleted']]
        # --- FIX ENDS HERE ---

        # Cluster by completion behavior patterns
        # Ensure n_clusters is not greater than the number of samples
        n_clusters_to_use = min(len(df.index), 3) # Max 3 clusters, but not more than available data
        if n_clusters_to_use < 1:
            print("⚠️ Not enough unique samples for clustering. Skipping KMeans.")
            return None
        
        kmeans = KMeans(n_clusters=n_clusters_to_use, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(data)

        os.makedirs(os.path.dirname(self.cluster_model_path), exist_ok=True)
        joblib.dump(kmeans, self.cluster_model_path)
        joblib.dump(encoder, 'models/cluster_encoder.pkl') # Save encoder

        print("✅ Task clustering model trained and saved")
        
        # Show completion rates per cluster
        cluster_completion = df.groupby('cluster')['isCompleted'].mean()
        print("Completion rates per cluster:")
        print(cluster_completion)
        
        return df

    def train_priority_model(self):
        """Use Linear Regression to analyze completion factors"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()

        existing_columns = [col for col in ['deadline', 'category', 'duration', 'is_flexible', 'isCompleted'] if col in df.columns]
        if existing_columns:
            df = df.dropna(subset=existing_columns)
            print(f"✅ Dropped NA values from columns: {existing_columns}")
        else:
            print("⚠️ None of the required columns exist in the DataFrame")
            return None # Cannot train without essential columns
               
        if df.empty or len(df) < 2:
            print("⚠️ Not enough data after dropping NA to train priority model.")
            return None

        # Feature engineering
        # Use a more robust way to calculate deadline_hours
        df['deadline_dt'] = pd.to_datetime(df['deadline'])
        current_time = pd.Timestamp.now()
        df['deadline_hours'] = (df['deadline_dt'] - current_time).dt.total_seconds() / 3600
        
        encoder = LabelEncoder()
        df['category_encoded'] = encoder.fit_transform(df['category'])

        # Use same features but predict completion probability
        X = df[['deadline_hours', 'duration', 'is_flexible', 'category_encoded']]
        y = df['isCompleted'].astype(float)  # Predict completion probability

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        os.makedirs(os.path.dirname(self.priority_model_path), exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "encoder": encoder
        }, self.priority_model_path)

        print("✅ Completion analysis model trained and saved!")
        print(f"R² Score: {model.score(X_test, y_test):.3f}")
        
        # Show feature importance for weight adjustments
        feature_importance = abs(model.coef_)
        print("Feature importance for completion:", feature_importance)
        
        return model

    def analyze_user_behavior(self):
        """Main ML analysis that calculates new weights based on user behavior"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()
        
        if len(df) < 5:  # Need minimum data for meaningful analysis
            print("⚠️ Not enough data for ML analysis to calculate new weights. Need at least 5 tasks.")
            return None
        
        # Analyze completion patterns
        completed = df[df['isCompleted'] == 1]
        
        # Ensure there's data to avoid division by zero
        if completed.empty:
            print("⚠️ No completed tasks found for ML behavior analysis. Skipping weight adjustment.")
            return None # Cannot meaningfully adjust weights without completed tasks
        
        # Calculate new weights based on actual behavior
        new_weights = {}
        
        # 1. Urgency weight adjustment
        # If tasks with higher initial scores are completed more often, increase urgency weight
        avg_score_completed = completed['score'].mean()
        avg_score_all = df['score'].mean()
        
        # Avoid division by zero
        urgency_ratio = avg_score_completed / avg_score_all if avg_score_all > 0 else 1.0
        # Adjust weight based on how much more "urgent" (by initial score) completed tasks were
        new_weights['urgency_weight'] = max(0.1, min(1.0, 0.35 * urgency_ratio))
        print(f"  - Urgency weight adjusted to: {new_weights['urgency_weight']:.3f} (based on score completion ratio: {urgency_ratio:.2f})")

        # 2. Category weight adjustments
        category_completion = df.groupby('category')['isCompleted'].mean().to_dict()
        for category in ['Work', 'Education']:
            if category in category_completion:
                completion_rate = category_completion[category]
                # If a category has a high completion rate, increase its weight
                new_weights[f'category_{category}'] = max(0.5, min(3.0, 2.0 * completion_rate))
                print(f"  - Category '{category}' weight adjusted to: {new_weights[f'category_{category}']:.3f} (completion rate: {completion_rate:.2f})")
            else:
                new_weights[f'category_{category}'] = new_weights.get(f'category_{category}', 1.0) # Keep default if no data
                print(f"  - No completion data for category '{category}', retaining default.")

        # Ensure 'category_other' is also handled
        if 'other' in category_completion:
            new_weights['category_other'] = max(0.5, min(3.0, 1.0 * category_completion['other']))
            print(f"  - Category 'other' weight adjusted to: {new_weights['category_other']:.3f}")
        else:
            new_weights['category_other'] = new_weights.get('category_other', 1.0)
            print("  - No completion data for category 'other', retaining default.")
        
        # 3. Importance weight adjustment  
        # Higher overall completion rate might imply users are good at prioritizing, so 'importance' works well.
        overall_completion_rate = df['isCompleted'].mean()
        new_weights['importance_weight'] = max(0.3, min(1.5, 0.7 * (1 + overall_completion_rate)))
        print(f"  - Importance weight adjusted to: {new_weights['importance_weight']:.3f} (overall completion rate: {overall_completion_rate:.2f})")
        
        # 4. Flexibility weight adjustment
        flexible_tasks_df = df[df['is_flexible'] == 1]
        flexible_completed_rate = flexible_tasks_df['isCompleted'].mean() if not flexible_tasks_df.empty else 0.5
        # If flexible tasks are completed often, increase its weight, suggesting flexibility is good
        new_weights['flexibility_weight'] = max(0.5, min(2.0, 1.0 * flexible_completed_rate))
        print(f"  - Flexibility weight adjusted to: {new_weights['flexibility_weight']:.3f} (flexible task completion rate: {flexible_completed_rate:.2f})")

        print(f"🔄 ML calculated new weights: {new_weights}")
        return new_weights

    def predict_priority(self, task_data):
        """Predict priority score for a new task"""
        # Load the latest state of the models and encoders
        try:
            model_data = joblib.load(self.priority_model_path)
            model = model_data["model"]
            scaler = model_data["scaler"]
            encoder = model_data["encoder"]
        except FileNotFoundError:
            print("⚠️ Priority model not found. Attempting to train before prediction.")
            # Check for sufficient data before training
            conn = sqlite3.connect(self.db_path)
            task_count = pd.read_sql_query("SELECT COUNT(*) FROM tasks", conn).iloc[0,0]
            conn.close()

            if task_count >= 5: # Minimum tasks for training
                self.train_cluster_model() # Ensure cluster encoder is available for potential category encoding
                self.train_priority_model() # Train priority model
            else:
                print(f"Insufficient tasks ({task_count}) for ML training. Cannot train priority model for prediction.")
                raise ValueError("Priority model not trained and insufficient data to train.")


            try: # Try loading again after training attempt
                model_data = joblib.load(self.priority_model_path)
                model = model_data["model"]
                scaler = model_data["scaler"]
                encoder = model_data["encoder"]
            except FileNotFoundError:
                 raise ValueError("Priority model still not trained after attempt. Cannot predict.")

        
        # Prepare features
        # Ensure 'deadline' is parsed correctly and calculate deadline_hours relative to now
        try:
            deadline_dt = datetime.strptime(task_data['deadline'], "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"Invalid deadline format: {task_data['deadline']}")

        deadline_hours = (deadline_dt - datetime.now()).total_seconds() / 3600
        
        # Handle new categories not seen during training
        try:
            category_encoded = encoder.transform([task_data['category']])[0]
        except ValueError:
            # If category is new, assign a default or the most frequent category's encoding
            print(f"New category '{task_data['category']}' encountered. Using a fallback encoding.")
            # A more sophisticated approach would be to get the mode of the training categories
            # For simplicity, let's just assign a default 'other' like encoding or 0
            category_encoded = 0 # Fallback to 0 or a specific 'other' category if you have it

        features = [[
            deadline_hours,
            task_data['duration'],
            task_data['is_flexible'],
            category_encoded
        ]]
        
        features_scaled = scaler.transform(features)
        predicted_score = model.predict(features_scaled)[0]
        
        # Clamp score between 0 and 100
        return max(0, min(100, predicted_score * 100)) # Scale to 0-100 if model predicts 0-1

    def get_task_clusters(self):
        """Get cluster assignments for all tasks"""
        if not os.path.exists(self.cluster_model_path):
            raise ValueError("Cluster model not trained yet. Call train_cluster_model() first.")
        
        kmeans = joblib.load(self.cluster_model_path)
        encoder = joblib.load('models/cluster_encoder.pkl') # Load encoder
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM tasks", conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame() # Return empty if no tasks

        # Ensure 'category' column exists
        if 'category' not in df.columns:
            print("⚠️ 'category' column not found in tasks table. Cannot get task clusters.")
            return pd.DataFrame()

        # Encode categories for clustering, handle unknown categories
        df['category_encoded'] = df['category'].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        
        # Filter out rows with unknown categories before prediction if -1 is used
        data = df[['duration', 'category_encoded', 'isCompleted']]
        df['cluster'] = kmeans.predict(data)
        
        return df
    
    def kmeans_analysis(self, df):
        print("🔍 K-Means Cluster Analysis (using internal logic or existing model)...")
        if df.empty or len(df) < 3:
            print("⚠️ Not enough data for K-Means analysis.")
            return {}
        
        try:
            kmeans = joblib.load(self.cluster_model_path)
            encoder = joblib.load('models/cluster_encoder.pkl')
            # Ensure 'category' column exists before encoding
            if 'category' not in df.columns:
                print("⚠️ 'category' column not found in tasks table for K-Means analysis.")
                return {}
            df['category_encoded'] = df['category'].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        except (FileNotFoundError, ValueError):
            print("Cluster model or encoder not found, training new one for this analysis.")
            df_clustered = self.train_cluster_model()
            if df_clustered is None: return {}
            df = df_clustered # Use the dataframe with clusters
            
            # Re-load for consistency if needed, or rely on df_clustered
            try:
                kmeans = joblib.load(self.cluster_model_path)
                encoder = joblib.load('models/cluster_encoder.pkl')
            except (FileNotFoundError, ValueError):
                 print("Failed to load freshly trained cluster model components for K-Means.")
                 return {}

        if 'category_encoded' not in df.columns:
             encoder = LabelEncoder()
             df['category_encoded'] = encoder.fit_transform(df['category'])

        data = df[['duration', 'category_encoded', 'isCompleted']]
        # Predict clusters using the (potentially new) data and model
        df['cluster'] = kmeans.predict(data)
        
        cluster_analysis = df.groupby('cluster').agg({
            'isCompleted': 'mean',
            'duration': 'mean',
            'category': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
        }).to_dict('index')
        
        print("🔍 K-Means Cluster Analysis Results:")
        for cluster_id, stats in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {stats['isCompleted']:.1%} completion, Avg duration: {stats['duration']:.0f}min, Common category: {stats['category']}")
        
        return cluster_analysis

    def linear_regression_analysis(self, df):
        print("📊 Linear Regression Analysis (using internal logic or existing model)...")
        if df.empty or len(df) < 2:
            print("⚠️ Not enough data for Linear Regression analysis.")
            return {}

        try:
            model_data = joblib.load(self.priority_model_path)
            model = model_data["model"]
            scaler = model_data["scaler"]
            encoder = model_data["encoder"]
        except FileNotFoundError:
            print("Priority model not found, training new one for this analysis.")
            model = self.train_priority_model()
            if model is None: return {}
            model_data = joblib.load(self.priority_model_path) # Reload to get scaler and encoder
            model = model_data["model"]
            scaler = model_data["scaler"]
            encoder = model_data["encoder"]

        # Ensure 'deadline' and 'category' columns exist
        if 'deadline' not in df.columns or 'category' not in df.columns:
            print("⚠️ Missing 'deadline' or 'category' column for Linear Regression analysis.")
            return {}

        df['deadline_dt'] = pd.to_datetime(df['deadline'])
        current_time = pd.Timestamp.now()
        df['deadline_hours'] = (df['deadline_dt'] - current_time).dt.total_seconds() / 3600

        if 'category_encoded' not in df.columns:
            df['category_encoded'] = df['category'].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0) # Fallback encoding

        X = df[['deadline_hours', 'duration', 'is_flexible', 'category_encoded']]
        X_scaled = scaler.transform(X) # Use the loaded scaler

        # Get feature importance (coefficients)
        feature_importance = abs(model.coef_)
        feature_names = ['urgency', 'duration', 'flexibility', 'category']
        
        print("📊 Linear Regression Feature Importance:")
        for name, importance in zip(feature_names, feature_importance):
            print(f"  {name}: {importance:.3f}")
        
        # Convert to weight adjustments
        weights = {}
        # These are illustrative adjustments, you'll want to fine-tune based on your model's coefficients
        weights['urgency_weight'] = max(0.1, min(1.0, 0.35 * (1 + feature_importance[0])))
        weights['importance_weight'] = max(0.3, min(1.5, 0.7 * (1 + feature_importance[3])))  # category importance
        weights['flexibility_weight'] = max(0.5, min(2.0, 1.0 * (1 + feature_importance[2])))
        
        return weights

    def combine_insights(self, clusters, weights):
        print("🎯 Combining insights from clusters and regression...")
        final_weights = weights.copy()
        
        # Use cluster insights to refine weights
        for cluster_id, cluster_info in clusters.items():
            completion_rate = cluster_info.get('isCompleted', 0)
            common_category = cluster_info.get('category')
            
            if common_category:
                # If a cluster (e.g., of a specific category) has high completion rate, boost its category weight
                if completion_rate > 0.7:  # E.g., 70% completion
                    category_key = f"category_{common_category}"
                    if category_key in final_weights:
                        final_weights[category_key] = min(3.0, final_weights[category_key] * 1.2)
                        print(f"  - Boosted {category_key} due to high completion in Cluster {cluster_id}")
                    elif common_category == 'Other' and 'category_other' in final_weights: # Handle 'Other' category specifically
                         final_weights['category_other'] = min(3.0, final_weights['category_other'] * 1.2)
                         print(f"  - Boosted category_other due to high completion in Cluster {cluster_id}")
        
        print("🎯 Final Combined Weights:", final_weights)
        return final_weights