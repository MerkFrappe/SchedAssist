
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
            db.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    id TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            db.commit()
            print("✅ Database initialized: tasks.db")
    
    # Also ensure app_state table exists even if database exists
    else:
        with app.app_context():
            db = get_db()
            db.execute("""
                CREATE TABLE IF NOT EXISTS app_state (
                    id TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            db.commit()
            print("✅ Ensured app_state table exists")
    
def get_optimization_state():
    """Check if optimization has been applied"""
    conn = get_db()
    state = conn.execute("SELECT value FROM app_state WHERE id = 'optimized'").fetchone()
    return state and state['value'] == 'true'

def set_optimization_state(optimized):
    """Set optimization state"""
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO app_state (id, value) 
        VALUES ('optimized', ?)
    """, ('true' if optimized else 'false',))
    conn.commit()
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
    
    urgent_count = len([t for t in tasks_for_calendar if t['score'] >= 70])
    important_count = len([t for t in tasks_for_calendar if t['score'] >= 40 and t['score'] < 70])
    normal_count = len([t for t in tasks_for_calendar if t['score'] >= 10 and t['score'] < 40])
    unessential_count = len([t for t in tasks_for_calendar if t['score'] < 10])        
    
    
    formatted_date = now.strftime("%B %d, %Y")
    # Calculate week number (simple approach, adjust if you need ISO week numbers)
    week_number = now.isocalendar()[1] 

    optimized = get_optimization_state()

    
    return render_template("Main.html",
    formatted_date=formatted_date,
    week_number=week_number, 
    optimized=optimized,     
    tasks=tasks_for_calendar,
    urgent_count=urgent_count,
    important_count=important_count,
    normal_count=normal_count,
    unessential_count=unessential_count,)


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
        SELECT id, title, category, deadline, duration, score, reminders, is_flexible
        FROM tasks
        WHERE isCompleted = 0
        ORDER BY score DESC
    """).fetchall()

    # Convert to dict and parse datetime
    tasks = []
    for row in tasks_raw:
        task = dict(row)
        try:
            deadline_dt = datetime.strptime(task["deadline"], "%Y-%m-%d %H:%M")
            duration_td = timedelta(minutes=int(task.get("duration", 60)))
            start_dt = deadline_dt - duration_td

            task["weekday"] = start_dt.strftime("%A")
            task["start_hour_int"] = start_dt.hour
            task["end_hour_int"] = deadline_dt.hour
            task["start_hour_str"] = start_dt.strftime('%H:%M')
            task["end_hour_str"] = deadline_dt.strftime('%H:%M')
            task['colspan'] = max(1, task['end_hour_int'] - task['start_hour_int'])
            
            # Parse reminders
            if task.get("reminders") and task["reminders"] not in (None, "", "null"):
                task["reminders"] = [r.strip() for r in task["reminders"].split(",") if r.strip()]
            else:
                task["reminders"] = []
                
            tasks.append(task)
        except Exception as e:
            print(f"⚠️ Error parsing task {task.get('title', 'N/A')}: {e}")

    # ==================== RESCHEDULE FUNCTIONS ====================
    def reschedule_flexible_tasks(tasks):
        """Reschedule flexible tasks with low scores to empty time slots"""
        # Initialize week schedule (6 AM to 10 PM, 16 hours)
        week_schedule = {
            day: [None] * 16 for day in [
                "Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday"
            ]
        }
        
        # First, place all non-flexible and high-priority tasks
        for task in tasks:
            if task.get('is_flexible', 0) == 0 or task.get('score', 0) >= 50:
                place_task_in_schedule(week_schedule, task)
        
        # Then, reschedule flexible low-priority tasks
        flexible_tasks = [t for t in tasks if t.get('is_flexible', 0) == 1 and t.get('score', 0) < 50]
        print(f"🔄 Found {len(flexible_tasks)} flexible low-priority tasks to reschedule")
        
        for task in flexible_tasks:
            if not reschedule_to_empty_slot(week_schedule, task):
                # If no empty slot today, move to tomorrow
                move_to_tomorrow(task)
                # Try to place it in the new day
                place_task_in_schedule(week_schedule, task)
        
        return week_schedule

    def place_task_in_schedule(week_schedule, task):
        """Place a task in its time slot if available"""
        try:
            day = task['weekday']
            start_hour = task['start_hour_int']
            duration_hours = max(1, (task['end_hour_int'] - task['start_hour_int']))
            
            # Check if slot is available
            can_place = True
            start_slot = start_hour - 6
            
            for h in range(start_slot, start_slot + duration_hours):
                if h < 0 or h >= 16 or week_schedule[day][h] is not None:
                    can_place = False
                    break
            
            if can_place:
                 # Update the actual deadline, not just display times
                    original_deadline = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
                    new_deadline = original_deadline.replace(
                    hour=task['start_hour_int'] + duration_hours,
                    minute=0
                )
                    task['deadline'] = new_deadline.strftime("%Y-%m-%d %H:%M")
                    print(f"✅ Placed '{task['title']}' on {day} at {start_hour}:00")
                    return True
            else:
                print(f"❌ Could not place '{task['title']}' on {day} at {start_hour}:00 - slot occupied")
                return False
            
        except Exception as e:
            print(f"Error placing task {task.get('title', 'Unknown')}: {e}")
            return False

    def reschedule_to_empty_slot(week_schedule, task):
        """Try to reschedule task to an empty slot in the same day"""
        try:
            original_day = task['weekday']
            duration_hours = max(1, (task['end_hour_int'] - task['start_hour_int']))
            
            # Try all time slots in the same day
            for start_slot in range(0, 16 - duration_hours + 1):
                # Check if this slot is empty
                slot_empty = True
                for h in range(start_slot, start_slot + duration_hours):
                    if week_schedule[original_day][h] is not None:
                        slot_empty = False
                        break
                
                if slot_empty:
                    # Place task in this empty slot
                    for h in range(start_slot, start_slot + duration_hours):
                        week_schedule[original_day][h] = task
                    
                    # Update task timing
                    new_start_hour = start_slot + 6
                    task['start_hour_int'] = new_start_hour
                    task['end_hour_int'] = new_start_hour + duration_hours
                    task['start_hour_str'] = f"{new_start_hour:02d}:00"
                    task['end_hour_str'] = f"{new_start_hour + duration_hours:02d}:00"
                    task['colspan'] = duration_hours
                    
                    print(f"✅ Rescheduled '{task['title']}' to {original_day} at {new_start_hour}:00")
                    return True
            
            print(f"❌ No empty slots found for '{task['title']}' on {original_day}")
            return False
            
        except Exception as e:
            print(f"Error rescheduling task {task.get('title', 'Unknown')}: {e}")
            return False

    def move_to_tomorrow(task):
        current_deadline = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
    # Move to next day
        next_day_date = current_deadline + timedelta(days=1)
        try:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            current_day_index = days.index(task['weekday'])
            next_day_index = (current_day_index + 1) % 7
            next_day = days[next_day_index]
            
            # Set to tomorrow at 9 AM (default)
            task['weekday'] = next_day
            task['start_hour_int'] = 9
            task['end_hour_int'] = 10  # Default 1-hour duration
            task['start_hour_str'] = "09:00"
            task['end_hour_str'] = "10:00"
            task['colspan'] = 1
            task['deadline'] = next_day_date.replace(hour=10, minute=0).strftime("%Y-%m-%d %H:%M")
            print(f"📅 Moved '{task['title']}' to {next_day} at 9:00 AM")
            
        except Exception as e:
            print(f"Error moving task to tomorrow {task.get('title', 'Unknown')}: {e}")
            # Fallback: keep original timing
            pass

    def save_optimized_schedule(optimized_schedule):
        """Save the optimized schedule back to the database with proper dates"""
        conn = get_db()
        print("🔍 DEBUG: Starting save_optimized_schedule")
        print(f"🔍 Schedule has {sum(len(slots) for slots in optimized_schedule.values())} slots")
        
        # Get current week dates
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())
        days_dates = {
            "Monday": start_of_week,
            "Tuesday": start_of_week + timedelta(days=1),
            "Wednesday": start_of_week + timedelta(days=2),
            "Thursday": start_of_week + timedelta(days=3),
            "Friday": start_of_week + timedelta(days=4),
            "Saturday": start_of_week + timedelta(days=5),
            "Sunday": start_of_week + timedelta(days=6),
        }
        
        updated_count = 0
        for day in optimized_schedule:
            for hour_slot in optimized_schedule[day]:
                if hour_slot is not None:
                    task = hour_slot
                    # Calculate new deadline based on day and end hour
                    original_deadline = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
                    new_deadline = original_deadlin.replace(
                        hour=task['end_hour_int'], 
                        minute=0
                        
                    )
                    
                    # Update the task in database
                    conn.execute("""
                        UPDATE tasks 
                        SET deadline = ?
                        WHERE id = ?
                    """, (new_deadline.strftime("%Y-%m-%d %H:%M"), task['id']))
                    updated_count += 1
        
        conn.commit()
        print(f"💾 Saved {updated_count} optimized tasks to database")
    # ==================== END OF RESCHEDULE FUNCTIONS ====================

    # Apply rescheduling
    optimized_schedule = reschedule_flexible_tasks(tasks)

    # Save to database
    save_optimized_schedule(optimized_schedule)

    set_optimization_state(True)
    
    # Convert schedule back to task list for display
    tasks_for_calendar = []
    seen_task_ids = set()
    
    for day in optimized_schedule:
        for task in optimized_schedule[day]:
            if task and task['id'] not in seen_task_ids:
                tasks_for_calendar.append(task)
                seen_task_ids.add(task['id'])

    # If no tasks were placed, fall back to original tasks
    if not tasks_for_calendar:
        tasks_for_calendar = tasks

    today = datetime.now()
    return render_template(
        "Main.html",
        tasks=tasks_for_calendar,
        formatted_date=today.strftime("%B %d, %Y"),
        week_number=today.isocalendar()[1],
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