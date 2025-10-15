import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.DataFrame([
    {"category": "Work", "duration": 120, "urgency": 8, "importance": 7, "isCompleted": 1},
    {"category": "Education", "duration": 90, "urgency": 3, "importance": 9, "isCompleted": 1},
    {"category": "Personal", "duration": 45, "urgency": 9, "importance": 4, "isCompleted": 0},
    {"category": "Work", "duration": 60, "urgency": 7, "importance": 8, "isCompleted": 1},
    {"category": "Education", "duration": 30, "urgency": 2, "importance": 10, "isCompleted": 1},
    {"category": "Work", "duration": 120, "urgency": 8, "importance": 7, "isCompleted": 1},
    {"category": "Work", "duration": 120, "urgency": 8, "importance": 7, "isCompleted": 0},

])


encoder = LabelEncoder()
data["category_encoded"] = encoder.fit_transform(data["category"])


X = data[["category_encoded", "duration", "urgency", "importance"]]
y = data["isCompleted"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

sample = pd.DataFrame([[encoder.transform(["Education"])[0], 90, 5, 8]],
                      columns=["category_encoded", "duration", "urgency", "importance"])
prob = model.predict_proba(sample)[0][1]

print(f"Predicted completion probability: {prob:.2f}")


category_counts = data["category"].value_counts()
focus_category = category_counts.idxmax()
print(f"Focus Category: {focus_category}")


avg_urgency = data["urgency"].mean()
if avg_urgency <= 4:
    print("User Type: Planner 🧠")
else:
    print("User Type: Crammer ⚡")
