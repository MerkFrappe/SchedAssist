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
    # Get the current date
    now = datetime.now()
    # Find the start of the current week (Monday)
    start_of_week = now - timedelta(days=now.weekday())
    # Find the end of the current week (Sunday)
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)

    raw_tasks = conn.execute("""
    SELECT id, title, category, deadline, reminders, score, created_at, duration
    FROM tasks
    WHERE deadline BETWEEN ? AND ?
    ORDER BY deadline ASC
    """, (start_of_week.strftime("%Y-%m-%d %H:%M"), end_of_week.strftime("%Y-%m-%d %H:%M"))).fetchall()

    tasks_for_calendar = []
    for row in raw_tasks:
        task = dict(row)
        try:
            deadline_dt = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
            
            # Calculate start time by subtracting duration from deadline
            # Duration is in minutes, so convert to timedelta
            duration_td = timedelta(minutes=task.get('duration', 60)) # Default to 60 minutes if not set
            start_dt = deadline_dt - duration_td

            task['weekday'] = start_dt.strftime('%A') # The day the task *starts*
            task['start_hour_int'] = start_dt.hour
            task['end_hour_int'] = deadline_dt.hour # The hour the task *ends* (exclusive for display)
            
            # For display purposes if needed, though the int versions are better for comparison
            task['start_hour_str'] = start_dt.strftime('%H:%M')
            task['end_hour_str'] = deadline_dt.strftime('%H:%M')

            tasks_for_calendar.append(task)
        except Exception as e:
            print(f"Error parsing deadline for task {task.get('id', 'N/A')}: {task['deadline']} → {e}")
        
    formatted_date = now.strftime("%B %d, %Y")
    # Calculate week number (simple approach, adjust if you need ISO week numbers)
    week_number = now.isocalendar()[1] 

    return render_template("Main.html", tasks=tasks_for_calendar, formatted_date=formatted_date, week_number=week_number)


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

    today_str = datetime.now().strftime("%Y-%m-%d")
    tasks_today = sum(1 for t in archived_tasks if t['deadline'].startswith(today_str))
    tasks_this_week = len(archived_tasks)

    today = datetime.now()
    formatted_date = today.strftime("%B %d, %Y")
    week_number = today.isocalendar()[1]

    return render_template(
        "Analytics.html",
         archived_tasks=archived_tasks,
        category_labels=labels,
        counts=counts,
        day_labels=day_labels,
        day_counts=day_counts,
        tasks_today=tasks_today,
        tasks_this_week=tasks_this_week,
        formatted_date=formatted_date, 
        week_number=week_number
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
        SELECT title, category, deadline, duration, score, reminders
        FROM tasks
        ORDER BY score DESC
    """).fetchall()
    conn.close()

    tasks_for_calendar = []  # Define it here first!

    for row in tasks_raw:
        task = dict(row)
        try:
            # Safely handle missing reminders
            if "reminders" not in task or task["reminders"] in (None, "", "null"):
                task["reminders"] = []
            else:
                # Convert comma-separated string to list
                task["reminders"] = [r.strip() for r in task["reminders"].split(",") if r.strip()]

            # Parse datetime
            deadline_dt = datetime.strptime(task["deadline"], "%Y-%m-%d %H:%M")
            duration_td = timedelta(minutes=int(task.get("duration", 60)))
            start_dt = deadline_dt - duration_td

            # Compute display attributes
            task["weekday"] = start_dt.strftime("%A")
            task["start_hour_int"] = start_dt.hour
            task["end_hour_int"] = deadline_dt.hour
            task["start_hour_str"] = start_dt.strftime("%H:%M")
            task["end_hour_str"] = deadline_dt.strftime("%H:%M")

            tasks_for_calendar.append(task)

        except Exception as e:
            print(f"⚠️ Error parsing deadline for {task.get('title', 'N/A')}: {e}")

    today = datetime.now()
    formatted_date = today.strftime("%B %d, %Y")
    week_number = today.isocalendar()[1]

    print("Optimized tasks prepared for Main:", tasks_for_calendar)

    return render_template(
        "Main.html",
        tasks=tasks_for_calendar,
        formatted_date=formatted_date,
        week_number=week_number,
        optimized=True
    )


@app.route('/mark_completed/<int:task_id>', methods=['POST'])
def mark_completed(task_id):
    conn = get_db()
    conn.execute("UPDATE tasks SET isCompleted = 1 WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    print(f"Task {task_id} marked as completed!")
    return "OK", 200 # Return a success status for AJAX

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    conn = get_db()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    print(f"Task {task_id} deleted!")
    return "OK", 200 # Return a success status for AJAX

@app.route("/get_task_hierarchy_modal_content")
def get_task_hierarchy_modal_content():
    conn = get_db()
    tasks = conn.execute("""
        SELECT id, title, category, deadline, duration, score
        FROM tasks
        WHERE isCompleted = 0
        ORDER BY score DESC, deadline ASC
    """).fetchall()
    conn.close()
    tasks = [dict(row) for row in tasks]
    # Render a partial template for the modal body
    return render_template("task_hierarchy_modal_content.html", tasks=tasks)

# App Entry Point
# -------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)