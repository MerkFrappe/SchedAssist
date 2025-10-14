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
                            isCompleted INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       """)
            db.commit()
            print(" Database initialized: tasks.db")
    

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("Get_Started.html")

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    tasks = conn.execute("""
        SELECT * FROM tasks
        ORDER BY score DESC
    """).fetchall()
    conn.close()
    tasks = [dict(row) for row in tasks]
    return render_template("dashboard.html", tasks=tasks, optimized=False)

@app.route("/Main")
def schedule():
  conn = get_db()
  raw_tasks = conn.execute("""
        SELECT title, category, deadline
        FROM tasks
        WHERE DATE(deadline) = DATE('now')
        ORDER BY deadline ASC
    """).fetchall()

  tasks = []
  for row in raw_tasks:
        task = dict(row)
        try:
            deadline_dt = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
            task['weekday'] = deadline_dt.strftime('%A')
            task['hour'] = deadline_dt.strftime('%H:00')       
            task['duration'] = int(task.get('duration', 1))  # fallback to 1 if missing

        # Generate all hours this task spans
            task['hours'] = [f"{h:02d}:00" for h in range(task['start_hour'], task['start_hour'] + task['duration'])]

            tasks.append(task)
        except Exception as e:
            print(f"Error parsing deadline: {task['deadline']} → {e}")
  print("Raw deadline:", task['deadline'])
  formatted_date = datetime.now().strftime("%B %d, %Y")
  week_number = (datetime.now().day - 1) // 7 + 1

  return render_template("Main.html", tasks=tasks, formatted_date=formatted_date, week_number=week_number)
@app.route("/Analytics")
def analytics():
    conn = get_db()

    # Get archived tasks
    archived_tasks = conn.execute("""
        SELECT * FROM tasks
        WHERE isCompleted = 1
        AND created_at >= DATE('now', '-7 days')
        ORDER BY created_at ASC
    """).fetchall()

    # Convert Row objects → dicts
    archived_tasks = [dict(row) for row in archived_tasks]

    # Get all tasks grouped by category
    category_counts = conn.execute("""
        SELECT category, COUNT(*) AS count
        FROM tasks
        GROUP BY category
    """).fetchall()

    completed_per_day = conn.execute("""
        SELECT DATE(created_at) AS date, COUNT(*) AS count
        FROM tasks
        WHERE isCompleted = 1
        AND created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY DATE(created_at) ASC
    """).fetchall()
    completed_per_day = [dict(row) for row in completed_per_day]
    category_counts = [dict(row) for row in category_counts]

    # Prepare data for charts
    day_labels = [row['date'] for row in completed_per_day]
    day_counts = [row['count'] for row in completed_per_day]

    labels = [row['category'] for row in category_counts]
    counts = [row['count'] for row in category_counts]

    return render_template(
        "Analytics.html",
        archived_tasks=archived_tasks,
        labels=labels,
        counts=counts,
        day_labels=day_labels,
        day_counts=day_counts
    )




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
               INSERT INTO tasks (title, deadline, duration, is_flexible, reminders, category, score, isCompleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               """, (title, deadline.strftime("%Y-%m-%d %H:%M"), duration_minutes, is_flexible, ",".join(reminders), category, score, 0))
    db.commit()

    print(f"Task '{title}' saved with score {score:.2f}")
    return redirect(url_for('dashboard'))
  
@app.route("/optimize_tasks", methods=["POST"])
def optimize_tasks():
    conn = get_db()
    tasks_raw = conn.execute("""
        SELECT title, category, deadline, duration, score
        FROM tasks
        ORDER BY score DESC
    """).fetchall()
    conn.close()

    tasks = []
    for row in tasks_raw:
        task = dict(row)
        try:
            deadline_dt = datetime.strptime(task["deadline"], "%Y-%m-%d %H:%M")
            task["weekday"] = deadline_dt.strftime("%A")
            task['start_hour'] = deadline_dt.hour
            task["duration"] = int(task.get("duration", 1))
            task['hours'] = [f"{h:02d}:00" for h in range(task['start_hour'], task['start_hour'] + task['duration'])]

            tasks.append(task)

        except Exception as e:
            print(f"⚠️ Error parsing deadline for {task['title']}: {e}")

    today = datetime.now()
    formatted_date = today.strftime("%B %d, %Y")
    week_number = (today.day - 1) // 7 + 1

    print("✅ Optimized tasks prepared for Main:", tasks)

    return render_template(
        "Main.html",
        tasks=tasks,
        formatted_date=formatted_date,
        week_number=week_number,
        optimized=True
    )

# App Entry Point
# -------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
