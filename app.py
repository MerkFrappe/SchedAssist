from flask import Flask, render_template, request, redirect, url_for, g
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)
DATABASE = "tasks.db"

# -------------------------
# Database Helper Functions
# -------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row  # allows dict-like access
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    """Creates the database if not exists"""
    if not os.path.exists(DATABASE):
        with app.app_context():
            db = get_db()
            db.execute("""
                       CREATE TABLE IF NOT EXISTS tasks (
                                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                            title TEXT NOT NULL,
                                                            deadline TEXT NOT NULL,
                                                            duration INTEGER,
                                                            is_flexible INTEGER,
                                                            reminders TEXT,
                                                            category TEXT,
                                                            score REAL,
                                                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       """)
            db.commit()
            print("✅ Database initialized: tasks.db")

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("Get_Started.html")

@app.route("/dashboard")
def dashboard():
    db = get_db()
    tasks = db.execute("SELECT * FROM tasks ORDER BY score DESC").fetchall()
    return render_template("dashboard.html", tasks=tasks)

@app.route("/Main")
def schedule():
    return render_template("Main.html")

@app.route("/Analytics")
def analytics():
    return render_template("Analytics.html")

@app.route('/submit', methods=['POST'])
def submit():
    title = request.form.get('title')
    hours = int(request.form.get("duration_hours", 0))
    minutes = int(request.form.get("duration_minutes", 0))
    category = request.form.get('category')

    reminder_offsets = {
        "5m": (timedelta(minutes=5), 5),
        "10m": (timedelta(minutes=10), 15),
        "15m": (timedelta(minutes=15), 10),
        "30m": (timedelta(minutes=30), 8),
        "1h": (timedelta(hours=1), 10),
        "3h": (timedelta(hours=3), 7),
        "1d": (timedelta(days=1), 5)
    }

    reminders = request.form.getlist("reminders[]")
    has_reminder = any(r != "none" for r in reminders)
    duration_minutes = hours * 60 + minutes

    # Deadline parsing
    try:
        deadline = datetime.strptime(request.form.get('deadline'), "%Y-%m-%dT%H:%M")
    except ValueError:
        return "Invalid date or time format.", 400

    now = datetime.now()
    remaining_time = (deadline - now).total_seconds() / 60

    urgency_score = 100 if remaining_time <= 0 else min((1 / remaining_time) * 100 + duration_minutes * 0.5, 100)

    # Category weighting
    category = category.capitalize()
    category_weight = 2 if category in ["Education", "Work"] else 1

    # Reminder scoring
    valid_reminders = [reminder_offsets[r] for r in reminders if r in reminder_offsets]
    reminder_value_total = sum(val for _, val in valid_reminders)
    importance = category_weight * reminder_value_total

    # Flexibility
    task_type = request.form.get('task_type', 'uninterrupted')
    is_flexible = 1 if task_type == 'flexible' else 0
    flexibility_weight = 1.0 if is_flexible else 1.2

    # Final priority score
    score = (urgency_score * 0.35) + (importance * 0.7) * flexibility_weight

    # Insert into database
    db = get_db()
    db.execute("""
               INSERT INTO tasks (title, deadline, duration, is_flexible, reminders, category, score)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               """, (title, deadline.strftime("%Y-%m-%d %H:%M"), duration_minutes, is_flexible, ",".join(reminders), category, score))
    db.commit()

    print(f"✅ Task '{title}' saved with score {score:.2f}")
    return redirect(url_for('dashboard'))

# -------------------------
# App Entry Point
# -------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
