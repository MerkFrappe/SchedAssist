from math import ceil
from models.ml_service import MLService
from flask import Flask, jsonify, render_template, request, redirect, url_for, g
from datetime import datetime, timedelta
import sqlite3
import os
import threading
import time


app = Flask(__name__)
DATABASE = "tasks.db"

absolute_db_path = os.path.abspath(DATABASE)
print(f"--- DEBUG: Flask is trying to use database at: {absolute_db_path} ---")

ml_service_instance = MLService(db_path=DATABASE) 

# -------------------------
# Database Helper Functions
# -------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        # Create tables if they don't exist
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
        db.execute("""
            CREATE TABLE IF NOT EXISTS ml_weights (
                weight_name TEXT PRIMARY KEY,
                weight_value REAL,
                confidence REAL DEFAULT 1.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert default weights if they don't exist
        db.execute("""
            INSERT OR IGNORE INTO ml_weights VALUES 
                ('urgency_weight', 0.35, 1.0, CURRENT_TIMESTAMP),
                ('importance_weight', 0.7, 1.0, CURRENT_TIMESTAMP),
                ('flexibility_weight', 1.0, 1.0, CURRENT_TIMESTAMP),
                ('category_Work', 2.0, 1.0, CURRENT_TIMESTAMP),
                ('category_Education', 2.0, 1.0, CURRENT_TIMESTAMP),
                ('category_other', 1.0, 1.0, CURRENT_TIMESTAMP)
        """)
        
        db.commit()
        print("✅ Database initialized: tasks.db")
        
        # Verify and fix schema
        verify_and_fix_schema()


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

def get_ml_weights():
    """Get current ML weights"""
    conn = get_db()
    weights = conn.execute("SELECT weight_name, weight_value FROM ml_weights").fetchall()
    return {row['weight_name']: row['weight_value'] for row in weights}

def update_ml_weights(new_weights):
    """Update ML weights (called by weekly ML job)"""
    conn = get_db()
    for weight_name, weight_value in new_weights.items():
        conn.execute("""
            UPDATE ml_weights 
            SET weight_value = ?, last_updated = CURRENT_TIMESTAMP
            WHERE weight_name = ?
        """, (weight_value, weight_name))
    conn.commit()
    print("✅ ML weights updated")

def verify_and_fix_schema():
    """Verify and fix any schema inconsistencies"""
    conn = get_db()
    
    try:
        # Check if isCompleted column exists
        columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
        column_names = [col[1] for col in columns]
        print("📋 Current task table columns:", column_names)
        
        # Fix isCompleted column if it doesn't exist
        if 'isCompleted' not in column_names:
            print("🔄 Adding missing isCompleted column...")
            conn.execute("ALTER TABLE tasks ADD COLUMN isCompleted INTEGER DEFAULT 0")
            conn.commit()
            print("✅ Added isCompleted column")
            
            # Update any existing tasks to have default value
            conn.execute("UPDATE tasks SET isCompleted = 0 WHERE isCompleted IS NULL")
            conn.commit()
            
    except Exception as e:
        print(f"❌ Error verifying schema: {e}")


def get_unified_notifications():
    """Get all notifications: reminders + urgency changes"""
    conn = get_db()
    now = datetime.now()
    notifications = []
    
    # First, let's verify the table structure
    try:
        # Test query to check column names
        test_columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
        column_names = [col[1] for col in test_columns]
        print("🔍 Task table columns:", column_names)
        
        # Use the correct column name (adjust based on what we find)
        completed_column = 'isCompleted'
        if 'isCompleted' not in column_names and 'iscompleted' in column_names:
            completed_column = 'iscompleted'
        elif 'isCompleted' not in column_names and 'completed' in column_names:
            completed_column = 'completed'
        
        print(f"🔧 Using column name: {completed_column}")
        
        # 1. Get upcoming reminders (triggered before task *start* time)
        tasks = conn.execute(f"""
            SELECT id, title, deadline, duration, reminders
            FROM tasks 
            WHERE {completed_column} = 0
        """).fetchall()

        for task in tasks:
            try:
        # Parse deadline and compute start time
                deadline_dt = datetime.strptime(task['deadline'], "%Y-%m-%d %H:%M")
                duration = task['duration'] if 'duration' in task.keys() and task['duration'] else 60
                duration = int(duration) # default 60 minutes if missing
                start_dt = deadline_dt - timedelta(minutes=duration)
                reminder_list = (task['reminders'] or "").split(",")

                for r in reminder_list:
                    r = r.strip()
                    if not r:
                        continue

                    amount_str = "".join([ch for ch in r if ch.isdigit()])
                    if not amount_str:
                        continue

                    amount = int(amount_str)
                    unit = "".join([ch for ch in r if ch.isalpha()])
                    if unit == "m":
                        offset = timedelta(minutes=amount)
                    elif unit == "h":
                        offset = timedelta(hours=amount)
                    elif unit == "d":
                        offset = timedelta(days=amount)
                    else:
                        continue  # skip unrecognized units

                    reminder_time = start_dt - offset

        # Check if reminder should trigger soon (before *start*, not before deadline)
                    if reminder_time <= now < reminder_time + timedelta(minutes=1):
                        notifications.append({
                    'type': 'reminder',
                    'task_id': task['id'],
                    'title': task['title'],
                    'message': f"Starting soon: {task['title']}",
                    'priority': 'medium',
                    'start_time': start_dt.strftime("%Y-%m-%d %H:%M"),
                    'deadline': task['deadline']
                })
            except Exception as e:
                print(f"⚠️ Error computing start time for task {task['title']}: {e}")
        
        # 2. Get urgent tasks (due in next 1 hour)
        urgent_threshold = now + timedelta(hours=1)
        urgent_tasks = conn.execute(f"""
            SELECT id, title, deadline, score
            FROM tasks 
            WHERE {completed_column} = 0 
            AND deadline BETWEEN ? AND ?
            AND score >= 70
        """, (now, urgent_threshold)).fetchall()
        
        for task in urgent_tasks:
            notifications.append({
                'type': 'urgency',
                'task_id': task['id'],
                'title': task['title'],
                'message': f"URGENT: {task['title']}",
                'priority': 'high',
                'deadline': task['deadline'],
                'score': task['score']
            })
        
        # 3. Get overdue tasks
        overdue_tasks = conn.execute(f"""
            SELECT id, title, deadline
            FROM tasks 
            WHERE {completed_column} = 0 
            AND deadline < ?
        """, (now,)).fetchall()
        
        for task in overdue_tasks:
            notifications.append({
                'type': 'overdue',
                'task_id': task['id'],
                'title': task['title'],
                'message': f"OVERDUE: {task['title']}",
                'priority': 'critical',
                'deadline': task['deadline']
            })
        
    except Exception as e:
        print(f"❌ Error in get_unified_notifications: {e}")
        # Fallback to basic query without isCompleted filter
        try:
            urgent_tasks = conn.execute("""
                SELECT id, title, deadline
                FROM tasks 
                WHERE deadline < ?
            """, (now,)).fetchall()
            
            for task in urgent_tasks:
                notifications.append({
                    'type': 'overdue',
                    'task_id': task['id'],
                    'title': task['title'],
                    'message': f"OVERDUE: {task['title']}",
                    'priority': 'critical',
                    'deadline': task['deadline']
                })
        except Exception as e2:
            print(f"❌ Even fallback query failed: {e2}")
    
    return notifications

# -------------------------
# Routes - REMOVE ALL conn.close() calls!
# -------------------------
@app.route("/")
def home():
    return render_template("Get_Started.html")

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    
    # Debug: print what we're getting from database
    print("🔄 Fetching tasks for dashboard...")
    
    # MODIFIED: Filter for incomplete tasks
    tasks = conn.execute("""
        SELECT * FROM tasks
        WHERE isCompleted = 0
        ORDER BY score DESC
    """).fetchall()
    
    print(f"📊 Found {len(tasks)} tasks in database")
    
    tasks = [dict(row) for row in tasks]
    
    # Debug print first few tasks
    for i, task in enumerate(tasks[:3]):
        print(f"  Task {i+1}: '{task.get('title', 'No title')}' - Score: {task.get('score', 'No score')}")
    
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
    WHERE isCompleted = 0  -- MODIFIED: Filter for incomplete tasks
    AND deadline BETWEEN ? AND ?
    ORDER BY deadline ASC
    """, (start_of_week.strftime("%Y-%m-%d %H:%M"), end_of_week.strftime("%Y-%m-%d %H:%M"))).fetchall()

    tasks_for_calendar = []
    for row in raw_tasks:
        task = dict(row)
        try:
            deadline_dt = datetime.strptime(task['deadline'], "%Y-%m-%dT%H:%M" if 'T' in task['deadline'] else "%Y-%m-%d %H:%M")
            
            duration_td = timedelta(minutes=task.get('duration', 60))
            start_dt = deadline_dt - duration_td

            task['weekday'] = start_dt.strftime('%A')
            task['start_hour_int'] = start_dt.hour
            task['end_hour_int'] = deadline_dt.hour
            task['start_minute_int'] = start_dt.minute
            task['end_minute_int'] = deadline_dt.minute
            task['start_hour_str'] = start_dt.strftime('%H:%M')
            task['end_hour_str'] = deadline_dt.strftime('%H:%M')

            if not task.get('reminders'):
                task['reminders'] = []

            tasks_for_calendar.append(task)

        except Exception as e:
            print(f"Error parsing deadline for task {task.get('id', 'N/A')}: {task['deadline']} → {e}")
    
    urgent_count = len([t for t in tasks_for_calendar if t['score'] >= 70])
    important_count = len([t for t in tasks_for_calendar if t['score'] >= 40 and t['score'] < 70])
    normal_count = len([t for t in tasks_for_calendar if t['score'] >= 10 and t['score'] < 40])
    unessential_count = len([t for t in tasks_for_calendar if t['score'] < 10])        
    
    formatted_date = now.strftime("%B %d, %Y")
    week_number = now.isocalendar()[1] 

    optimized = get_optimization_state()

    advice_text = generate_advice_from_weights()


    return render_template("Main.html",
        formatted_date=formatted_date,
        week_number=week_number, 
        optimized=optimized,     
        tasks=tasks_for_calendar,
        urgent_count=urgent_count,
        important_count=important_count,
        normal_count=normal_count,
        unessential_count=unessential_count,
        advice_text=advice_text 
        )

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

    analytics_summary = generate_analytics_summary()
    weights = get_ml_weights()


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
        week_number=week_number,
        analytics_summary=analytics_summary,
        weights=weights

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
        deadline_str = request.form.get('deadline')
        deadline = datetime.strptime(deadline_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return "Invalid date or time format.", 400

    task_type = request.form.get('task_type', 'uninterrupted')
    is_flexible = 1 if task_type == 'flexible' else 0

    task_data_for_ml = {
        'deadline': deadline_str,
        'duration': duration_minutes,
        'is_flexible': is_flexible,
        'category': category
    }
    print(f"🔍 DEBUG: Task data for ML prediction: {task_data_for_ml}")

    try:
        # Predict the priority score using the ML service
        score = ml_service_instance.predict_priority(task_data_for_ml)
        print(f"🎯 DEBUG: ML predicted score for '{title}': {score:.2f}")
    except ValueError as e:
        print(f"⚠️ ERROR: ML model not trained, falling back to default score. {e}")
        # Fallback to a default scoring if ML model isn't ready
        now = datetime.now()
        remaining_time = (deadline - now).total_seconds() / 60
        urgency_score = 100 if remaining_time <= 0 else min((1 / remaining_time) * 100 + duration_minutes * 0.5, 100)
        
        # Default weights for fallback
        default_weights = get_ml_weights() 
        category_key = f"category_{category.capitalize()}"
        category_weight = default_weights.get(category_key, default_weights['category_other'])
        valid_reminders = [reminder_offsets[r] for r in reminders if r in reminder_offsets]
        reminder_value_total = sum(val for _, val in valid_reminders)
        importance = category_weight * reminder_value_total
        flexibility_weight_val = default_weights['flexibility_weight'] if is_flexible else 1.2
        score = (urgency_score * default_weights['urgency_weight']) + (importance * default_weights['importance_weight']) * flexibility_weight_val

    # Insert into database
    print(f"🔧 DEBUG: Task '{title}' - is_flexible: {is_flexible}, task_type: '{task_type}'")
    
    db = get_db()
    db.execute("""
               INSERT INTO tasks (title, deadline, duration, is_flexible, reminders, category, score, isCompleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               """, (title, deadline.strftime("%Y-%m-%d %H:%M"), duration_minutes, is_flexible, ",".join(reminders), category, score, 0))
    db.commit()

    print(f"Task '{title}' saved with score {score:.2f}, flexible: {is_flexible}")

    # --- NEW: Check task count and trigger ML training ---
    task_count = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    if task_count >= 30:
        print(f"Total tasks ({task_count}) >= 30. Triggering ML model training.")
        train_ml() # Call the ML training function
    # --- END NEW ---

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

    print("🔍 DEBUG - Tasks from database:")
    for task in tasks_raw:
        print(f"  - '{task['title']}': is_flexible={task['is_flexible']}, score={task['score']}")

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
            task["start_minute_int"] = start_dt.minute
            task["end_minute_int"] = deadline_dt.minute
            task["start_hour_str"] = f"{start_dt.hour:02d}:{start_dt.minute:02d}"
            task["end_hour_str"] = f"{deadline_dt.hour:02d}:{deadline_dt.minute:02d}"
            
            # Calculate colspan based on 30-minute slots
            total_duration_minutes = task.get("duration", 60)
            task['colspan'] = max(1, ceil(total_duration_minutes / 15))
            #flag
            # Parse reminders
            if not task.get('reminders'):
                task['reminders'] = []
                
            tasks.append(task)
        except Exception as e:
            print(f"⚠️ Error parsing task {task.get('title', 'N/A')}: {e}")

    # ==================== RESCHEDULE FUNCTIONS ====================

    # Constants for schedule mapping
    SCHEDULE_START_HOUR = 6  # 6 AM
    SCHEDULE_END_HOUR = 22   # 10 PM
    SLOT_DURATION_MINUTES = 15
    TOTAL_DAILY_SLOTS = (SCHEDULE_END_HOUR - SCHEDULE_START_HOUR) * (60 // SLOT_DURATION_MINUTES)  # 32 slots

    def get_slot_index(hour, minute):
        """Converts hour and minute to slot index (0-31)"""
        if hour < SCHEDULE_START_HOUR or hour >= SCHEDULE_END_HOUR:
            return -1
        total_minutes = (hour - SCHEDULE_START_HOUR) * 60 + minute
        return total_minutes // SLOT_DURATION_MINUTES

    def get_time_from_slot_index(slot_index):
        """Converts slot index back to hour and minute"""
        total_minutes = slot_index * SLOT_DURATION_MINUTES
        hour = SCHEDULE_START_HOUR + (total_minutes // 60)
        minute = total_minutes % 60
        return hour, minute

    def get_num_slots_for_duration(duration_minutes):
        """Calculates number of slots needed for duration"""
        return max(1, ceil(duration_minutes / SLOT_DURATION_MINUTES))

    def initialize_schedule():
        """Initialize empty schedule for the week"""
        return {
            day: [None] * TOTAL_DAILY_SLOTS for day in [
                "Monday", "Tuesday", "Wednesday", "Thursday", 
                "Friday", "Saturday", "Sunday"
            ]
        }

    def can_place_task(schedule, day, start_slot, num_slots):
        """Check if task can be placed at given position"""
        if start_slot < 0 or start_slot + num_slots > TOTAL_DAILY_SLOTS:
            return False
        
        for slot in range(start_slot, start_slot + num_slots):
            if schedule[day][slot] is not None:
                return False
        return True

    def place_task(schedule, task, day, start_slot):
        """Place task at specified position"""
        num_slots = task['num_slots']
        
        # Update task timing
        new_start_hour, new_start_minute = get_time_from_slot_index(start_slot)
        new_end_hour, new_end_minute = get_time_from_slot_index(start_slot + num_slots)
        
        task['weekday'] = day
        task['start_slot_index'] = start_slot
        task['start_hour_int'] = new_start_hour
        task['start_minute_int'] = new_start_minute
        task['end_hour_int'] = new_end_hour
        task['end_minute_int'] = new_end_minute
        task['start_hour_str'] = f"{new_start_hour:02d}:{new_start_minute:02d}"
        task['end_hour_str'] = f"{new_end_hour:02d}:{new_end_minute:02d}"
        
        # Place in schedule
        for slot in range(start_slot, start_slot + num_slots):
            schedule[day][slot] = task
        
        return True

    def find_empty_slot(schedule, task, preferred_day=None, preferred_slot=None):
        """
        Smarter function to find an empty slot.
        1. Tries to place the task at the preferred day and time.
        2. If that fails, searches the rest of the preferred day.
        3. If that fails, searches all other days.
        """
        num_slots = task['num_slots']
        all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Reorder days to start with the preferred one
        if preferred_day:
            days_to_check = [preferred_day] + [day for day in all_days if day != preferred_day]
        else:
            days_to_check = all_days

        for day in days_to_check:
            start_search_index = 0
            # On the preferred day, start searching from the preferred slot
            if day == preferred_day and preferred_slot is not None:
                if can_place_task(schedule, day, preferred_slot, num_slots):
                    return day, preferred_slot # Ideal spot found!
                # If ideal spot is taken, search from that point forward
                start_search_index = preferred_slot + 1

            # Search the rest of the day (or the whole day if not preferred)
            for start_slot in range(start_search_index, TOTAL_DAILY_SLOTS - num_slots + 1):
                if can_place_task(schedule, day, start_slot, num_slots):
                    return day, start_slot # Found a spot on this day

            # If it's the preferred day and we still haven't found a spot,
            # we should also check the slots *before* the preferred time as a last resort.
            if day == preferred_day and preferred_slot is not None:
                for start_slot in range(0, preferred_slot):
                    if can_place_task(schedule, day, start_slot, num_slots):
                        return day, start_slot

        # If no slot was found on any day
        return None, None
    
    def reschedule_flexible_tasks(all_tasks):
        """Main rescheduling function"""
        schedule = initialize_schedule()
        weights = get_ml_weights()
        
        adaptive_threshold = 50 * (weights.get('importance_weight', 0.7) / 0.7)

        # Separate tasks by type
        fixed_tasks = [t for t in all_tasks if t.get('is_flexible', 0) == 0 or t.get('score', 0) >= adaptive_threshold]
        flexible_tasks = [t for t in all_tasks if t.get('is_flexible', 0) == 1 and t.get('score', 0) < adaptive_threshold] 
        
        print(f"📊 Task breakdown: {len(fixed_tasks)} fixed/high-priority, {len(flexible_tasks)} flexible/low-priority")
        
        # --- (The 'fixed_tasks' processing part remains unchanged) ---
        placed_tasks = []
        for task in fixed_tasks:
            # ... (no changes needed here) ...
            deadline_dt = datetime.strptime(task["deadline"], "%Y-%m-%d %H:%M")
            start_dt = deadline_dt - timedelta(minutes=task.get("duration", 60))
            
            original_day = start_dt.strftime('%A')
            original_slot = get_slot_index(start_dt.hour, start_dt.minute)
            task['num_slots'] = get_num_slots_for_duration(task.get("duration", 60))
            
            if original_slot >= 0 and can_place_task(schedule, original_day, original_slot, task['num_slots']):
                place_task(schedule, task, original_day, original_slot)
                placed_tasks.append(task['id'])
                print(f"✅ Fixed task placed: '{task['title']}' at {original_day} {task['start_hour_str']}")
            else:
                found_day, found_slot = find_empty_slot(schedule, task, original_day, original_slot)
                if found_day and found_slot is not None:
                    place_task(schedule, task, found_day, found_slot)
                    placed_tasks.append(task['id'])
                    print(f"🔄 Fixed task moved: '{task['title']}' to {found_day} {task['start_hour_str']}")
                else:
                    print(f"❌ Could not place fixed task: '{task['title']}'")

        # --- MODIFICATION IS HERE ---
        # Process flexible tasks
        for task in flexible_tasks:
            if task['id'] in placed_tasks:
                continue
                
            # Calculate original position
            deadline_dt = datetime.strptime(task["deadline"], "%Y-%m-%d %H:%M")
            start_dt = deadline_dt - timedelta(minutes=task.get("duration", 60))
            
            original_day = start_dt.strftime('%A')
            original_slot = get_slot_index(start_dt.hour, start_dt.minute) # This is the user's preferred slot
            task['num_slots'] = get_num_slots_for_duration(task.get("duration", 60))
            
            # Use the NEW, smarter find_empty_slot function
            found_day, found_slot = find_empty_slot(schedule, task, original_day, original_slot)
            
            if found_day and found_slot is not None:
                place_task(schedule, task, found_day, found_slot)
                placed_tasks.append(task['id'])
                
                # Your logging logic remains perfect
                if found_day == original_day and found_slot == original_slot:
                    print(f"✅ Flexible task placed: '{task['title']}' at its original time {found_day} {task['start_hour_str']}")
                else:
                    print(f"🔄 Flexible task rescheduled: '{task['title']}' to {found_day} {task['start_hour_str']}")
            else:
                print(f"❌ Could not place flexible task: '{task['title']}'")
        
        
        return schedule
    
    def save_optimized_schedule(optimized_schedule):
        """Save the optimized schedule to database"""
        conn = get_db()
        updated_count = 0
        processed_task_ids = set()
        
        # Get current week dates
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        
        for day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            day_index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_name)
            day_date = start_of_week + timedelta(days=day_index)
            
            for slot in optimized_schedule[day_name]:
                if slot and slot['id'] not in processed_task_ids:
                    task = slot
                    processed_task_ids.add(task['id'])
                    
                    try:
                        # Create new deadline using the task's end time on the correct date
                        new_deadline = day_date.replace(
                            hour=task['end_hour_int'],
                            minute=task['end_minute_int'],
                            second=0,
                            microsecond=0
                        )
                        
                        conn.execute("""
                            UPDATE tasks 
                            SET deadline = ?
                            WHERE id = ?
                        """, (new_deadline.strftime("%Y-%m-%d %H:%M"), task['id']))
                        updated_count += 1
                        
                    except Exception as e:
                        print(f"Error saving task {task.get('title', 'Unknown')}: {e}")
        
        conn.commit()
        print(f"💾 Saved {updated_count} optimized tasks to database")

    # ==================== APPLY OPTIMIZATION ====================
    
    print("🔄 Starting task optimization...")
    
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
        print("⚠️ No tasks could be optimized, using original schedule")

    urgent_count = len([t for t in tasks_for_calendar if t['score'] >= 70])
    important_count = len([t for t in tasks_for_calendar if t['score'] >= 40 and t['score'] < 70])
    normal_count = len([t for t in tasks_for_calendar if t['score'] >= 10 and t['score'] < 40])
    unessential_count = len([t for t in tasks_for_calendar if t['score'] < 10])

    today = datetime.now()
    print("✅ Optimization completed successfully!")
    
    return render_template(
        "Main.html",
        tasks=tasks_for_calendar,
        formatted_date=today.strftime("%B %d, %Y"),
        week_number=today.isocalendar()[1],
        optimized=True,
        urgent_count=urgent_count,
        important_count=important_count,
        normal_count=normal_count,
        unessential_count=unessential_count
    )

@app.route('/mark_completed/<int:task_id>', methods=['POST'])
def mark_completed(task_id):
    conn = get_db()
    conn.execute("UPDATE tasks SET isCompleted = 1 WHERE id = ?", (task_id,))
    conn.commit()
    return "OK", 200

@app.route('/delete_task/<int:task_id>', methods=['POST'])
def delete_task(task_id):
    conn = get_db()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return "OK", 200

@app.route("/get_task_hierarchy_modal_content")
def get_task_hierarchy_modal_content():
    conn = get_db()
    tasks = conn.execute("""
        SELECT id, title, category, deadline, duration, score
        FROM tasks
        WHERE isCompleted = 0
        ORDER BY score DESC, deadline ASC
    """).fetchall()
    tasks = [dict(row) for row in tasks]
    return render_template("task_hierarchy_modal_content.html", tasks=tasks)

@app.route("/train_ml", methods=["POST", "GET"]) # Added GET for manual trigger/testing
def train_ml():
    print("🔄 Triggering ML model training and weight analysis...")
    try:
        # We need to make sure there's enough data for training
        db = get_db()
        task_count = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        if task_count < 5: # MLService needs at least 5 tasks for analysis
            print(f"⚠️ Not enough tasks ({task_count}) for ML analysis. Skipping training.")
            return jsonify({"status": "insufficient_data", "message": "Not enough tasks to train ML models (need at least 5)."}), 200

        ml_service_instance.train_cluster_model()
        ml_service_instance.train_priority_model()
        
        new_weights = ml_service_instance.analyze_user_behavior()
        
        if new_weights:
            update_ml_weights(new_weights)
            print("✅ ML training and weight update successful!")
            return jsonify({"status": "success", "new_weights": new_weights})
        else:
            print("⚠️ ML analysis returned no new weights (insufficient data or error).")
            return jsonify({"status": "insufficient_data", "message": "Could not determine new weights."})
    except Exception as e:
        print(f"❌ ERROR during ML training or analysis: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route("/debug_weights")
def debug_weights():
    weights = get_ml_weights()
    return jsonify(weights)

@app.route("/api/notifications")
def api_notifications():
    """Endpoint for frontend to get all notifications"""
    notifications = get_unified_notifications()
    return jsonify(notifications)

@app.route("/debug_tasks")
def debug_tasks():
    """Debug route to see all tasks in database"""
    conn = get_db()
    
    # Get all tasks
    all_tasks = conn.execute("SELECT * FROM tasks").fetchall()
    tasks_list = [dict(task) for task in all_tasks]
    
    # Get column info
    columns = conn.execute("PRAGMA table_info(tasks)").fetchall()
    column_info = [dict(col) for col in columns]
    
    return jsonify({
        "total_tasks": len(tasks_list),
        "columns": column_info,
        "tasks": tasks_list
    })

@app.route("/test_ml_prediction", methods=["POST"])
def test_ml_prediction():
    """Test ML prediction with a sample task"""
    try:
        # Create a sample task for testing
        sample_task = {
            'deadline': (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M"),
            'duration': 60,
            'is_flexible': 0,
            'category': 'Work'
        }
        
        # Get ML prediction
        predicted_score = ml_service_instance.predict_priority(sample_task)
        
        return jsonify({
            "status": "success",
            "sample_task": sample_task,
            "predicted_score": round(predicted_score, 2),
            "current_weights": get_ml_weights()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route("/ml_analysis")
def ml_analysis():
    """Enhanced ML analysis for the analytics dashboard"""
    conn = get_db()
    
    # Get completion patterns by category
    category_patterns = conn.execute("""
        SELECT 
            category,
            AVG(CASE WHEN isCompleted = 1 THEN 1.0 ELSE 0.0 END) as completion_rate,
            AVG(score) as avg_score,
            COUNT(*) as task_count
        FROM tasks
        GROUP BY category
    """).fetchall()
    
    # Get completion by priority groups
    priority_stats = conn.execute("""
        SELECT 
            CASE 
                WHEN score >= 70 THEN 'High Priority'
                WHEN score >= 40 THEN 'Medium Priority' 
                ELSE 'Low Priority'
            END as priority_group,
            AVG(CASE WHEN isCompleted = 1 THEN 1.0 ELSE 0.0 END) as completion_rate,
            COUNT(*) as count
        FROM tasks
        GROUP BY priority_group
    """).fetchall()
    
    # Get flexibility patterns
    flexibility_stats = conn.execute("""
        SELECT 
            is_flexible,
            AVG(CASE WHEN isCompleted = 1 THEN 1.0 ELSE 0.0 END) as completion_rate,
            COUNT(*) as count
        FROM tasks
        GROUP BY is_flexible
    """).fetchall()
    
    return jsonify({
        "task_patterns": [dict(pattern) for pattern in category_patterns],
        "completion_stats": [dict(stat) for stat in priority_stats],
        "flexibility_stats": [dict(stat) for stat in flexibility_stats],
        "current_weights": get_ml_weights()
    })

@app.route("/api/current_tasks")
def api_current_tasks():
    """Get current tasks for ML notifications"""
    conn = get_db()
    
    # Get tasks for the current week
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    # Only select columns that actually exist in the database
    tasks = conn.execute("""
        SELECT id, title, category, deadline, score, isCompleted, duration
        FROM tasks
        WHERE deadline BETWEEN ? AND ?
        ORDER BY deadline ASC
    """, (start_of_week.strftime("%Y-%m-%d %H:%M"), end_of_week.strftime("%Y-%m-%d %H:%M"))).fetchall()
    
    tasks_list = []
    for task in tasks:
        task_dict = dict(task)  # Convert to dict to use .get() method
        
        try:
            # Parse deadline and calculate start/end times
            deadline_dt = datetime.strptime(task_dict['deadline'], "%Y-%m-%d %H:%M")
            task_dict['deadline'] = deadline_dt.isoformat()
            
            # Calculate start time from duration
            duration_minutes = task_dict.get('duration', 60)
            start_dt = deadline_dt - timedelta(minutes=duration_minutes)
            
            # Add calculated fields for frontend
            task_dict['start_hour_int'] = start_dt.hour
            task_dict['end_hour_int'] = deadline_dt.hour
            
        except Exception as e:
            print(f"⚠️ Error computing start time for task {task_dict.get('title', 'Unknown')}: {e}")
            # Set default values if calculation fails
            task_dict['start_hour_int'] = 9
            task_dict['end_hour_int'] = 10
            
        tasks_list.append(task_dict)
    
    return jsonify(tasks_list)

def generate_advice_from_weights():
    """Generate personalized advice text from current ML weights."""
    weights = get_ml_weights()
    if not weights:
        return "Not enough data yet for personalized advice."

    advice = []

    # Get current thresholds or use defaults
    urgency_threshold = 1.0
    importance_threshold = 1.0
    flexibility_threshold = 0.8
    category_work_threshold = 1.5
    category_education_threshold = 1.5
    duration_threshold_pref_short = 1.2
    duration_threshold_pref_long = 0.8

    # Using confidence if available to make advice more nuanced
    # For now, we'll use a simplified check
    
    if weights.get('flexibility_weight', 1.0) < flexibility_threshold:
        advice.append("You tend to delay flexible tasks — try scheduling them earlier.")
    if weights.get('urgency_weight', 1.0) > urgency_threshold:
        advice.append("You often rush to finish tasks close to their deadlines — set earlier reminders.")
    if weights.get('category_Work', 1.0) > category_work_threshold:
        advice.append("Work tasks dominate your schedule — keep an eye on rest and study balance.")
    if weights.get('category_Education', 1.0) > category_education_threshold:
        advice.append("You prioritize education tasks effectively — keep it up!")
    if weights.get('duration_weight', 1.0) > duration_threshold_pref_short:
        advice.append("You seem to prefer shorter tasks. Consider breaking down larger tasks.")
    elif weights.get('duration_weight', 1.0) < duration_threshold_pref_long:
        advice.append("You handle long tasks well. Ensure you schedule sufficient focus time.")

    if not advice:
        advice.append("Your task patterns look well-balanced. Stay consistent!")

    return " ".join(advice)


def generate_analytics_summary():
    """Simple decision-based analytics summary"""
    weights = get_ml_weights()
    
    if not weights:
        return {"summary": "No analytics data available yet.", "weights": {}}
    
    trends = []
    
    # Simple if-else decisions (much cleaner)
    urgency = weights.get('urgency_weight', 0.35)
    importance = weights.get('importance_weight', 0.7)
    flexibility = weights.get('flexibility_weight', 1.0)
    work = weights.get('category_Work', 1.0)
    education = weights.get('category_Education', 1.0)
    duration = weights.get('duration_weight', 1.0)  # Add duration
    
    # Urgency decisions
    if urgency > 1.0:
        trends.append("⚡ Deadline-driven: You're focusing on time-sensitive tasks")
    elif urgency < 0.5:
        trends.append("⏳ Even pace: You're spreading attention across deadlines")
    
    # Importance decisions  
    if importance > 1.0:
        trends.append("🎯 High-value focus: Prioritizing important tasks")
    elif importance < 0.5:
        trends.append("📊 Balanced value: Importance isn't driving your schedule")
    
    # Flexibility decisions
    if flexibility < 0.8:
        trends.append("📅 Structured: You're sticking to fixed-time tasks")
    elif flexibility > 1.2:
        trends.append("🔄 Adaptive: Great at fitting flexible tasks in gaps")
    
    # Category decisions
    if work > education + 0.3:
        trends.append("💼 Work focus: Professional tasks dominating")
    elif education > work + 0.3:
        trends.append("📚 Study mode: Academic tasks are priority")
    
    # Duration decisions (NEW - simple)
    if duration > 1.2:
        trends.append("⏱️ Quick tasks: Preferring shorter, faster tasks")
    elif duration < 0.8:
        trends.append("🧠 Deep work: Focusing on longer, complex tasks")
    
    return {
        "trends": trends,
        "weights": weights,
        "summary": " | ".join(trends) if trends else "No clear patterns yet"
    }



@app.route("/api/advice")
def api_advice():
    return jsonify({"advice": generate_advice_from_weights()})

def background_ml_training():
    """Background thread for ML training - runs monthly and scans last 30 days tasks"""
    # Use 30 days in seconds for approximately a month
    MONTHLY_INTERVAL_SECONDS = 30 * 24 * 60 * 60
    while True:
        time.sleep(MONTHLY_INTERVAL_SECONDS)  
        with app.app_context():
            # Scan for tasks in the last 30 days
            now = datetime.now()
            start_of_period = now - timedelta(days=30)
            
            db = get_db()
            recent_task_count = db.execute("""
                SELECT COUNT(*) FROM tasks 
                WHERE created_at BETWEEN ? AND ?
            """, (start_of_period.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S"))).fetchone()[0]
            
            if recent_task_count >= 5:  # Train if 5+ tasks in the last 30 days
                print(f"🔄 Monthly ML training triggered - {recent_task_count} tasks in last 30 days")
                train_ml()
            else:
                print(f"📊 Monthly ML training skipped: Only {recent_task_count} tasks in last 30 days")
 

@app.route('/update-task', methods=['POST'])
def update_task():
    """Handle task updates"""
    try:
        task_id = int(request.form.get('task_id'))
        title = request.form.get('title')
        deadline_str = request.form.get('deadline')
        duration_hours = int(request.form.get('duration_hours', 0))
        duration_minutes = int(request.form.get('duration_minutes', 0))
        task_type = request.form.get('task_type')
        category = request.form.get('category')
        reminders = request.form.getlist('reminders[]')
        
        # Parse deadline
        deadline = datetime.fromisoformat(deadline_str)
        
        # Calculate duration display
        duration_display = f"{duration_hours}h {duration_minutes}m"
        total_duration_minutes = (duration_hours * 60) + duration_minutes
        
        # Load tasks and find the one to update
        tasks = get_db()
        task_index = next((i for i, t in enumerate(tasks) if t['id'] == task_id), None)
        
        if task_index is not None:
            # Update task
            tasks[task_index].update({
                'title': title,
                'deadline': deadline,
                'duration': duration_display,
                'duration_minutes': total_duration_minutes,
                'is_flexible': task_type == 'flexible',
                'category': category,
                'reminders': reminders,
                'updated_at': datetime.now()
            })
            
            submit(tasks)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Task not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# App Entry Point
# -------------------------
if __name__ == "__main__":
    init_db()
    os.makedirs('models', exist_ok=True) 

    ml_thread = threading.Thread(target=background_ml_training, daemon=True)
    ml_thread.start()

    app.run(debug=True)