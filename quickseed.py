import sqlite3
from datetime import datetime, timedelta

DATABASE = "tasks.db"

def init_database():
    """Initialize the database with tables"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    # Create tables
    c.execute("""
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
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS app_state (
            id TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS ml_weights (
            weight_name TEXT PRIMARY KEY,
            weight_value REAL,
            confidence REAL DEFAULT 1.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert default weights
    c.execute("""
        INSERT OR IGNORE INTO ml_weights VALUES 
            ('urgency_weight', 0.35, 1.0, CURRENT_TIMESTAMP),
            ('importance_weight', 0.7, 1.0, CURRENT_TIMESTAMP),
            ('flexibility_weight', 1.0, 1.0, CURRENT_TIMESTAMP),
            ('category_Work', 2.0, 1.0, CURRENT_TIMESTAMP),
            ('category_Education', 2.0, 1.0, CURRENT_TIMESTAMP),
            ('category_Personal', 1.0, 1.0, CURRENT_TIMESTAMP)
    """)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized!")

def seed_tasks():
    """Seed the database with sample tasks"""
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    tasks =  [
    # ========== EDUCATION TASKS (Almost Perfect Completion) ==========
    ("Study for Calculus final", "2025-10-17 19:00", 180, 0, "1d,2h", "Education", 95, 1),
    ("Write 10-page research paper", "2025-10-18 20:00", 240, 0, "2d", "Education", 92, 1),
    ("Complete Physics lab report", "2025-10-19 16:00", 120, 0, "3h", "Education", 88, 1),
    ("Prepare Chemistry presentation", "2025-10-20 18:00", 150, 0, "1d", "Education", 90, 1),
    ("Read 5 chapters for Literature", "2025-10-21 21:00", 180, 1, "1h", "Education", 85, 1),
    ("Solve advanced Math problems", "2025-10-22 15:00", 120, 0, "2h", "Education", 89, 1),
    ("Research thesis topic", "2025-10-23 19:00", 180, 1, "1d", "Education", 93, 1),
    ("Study group - exam prep", "2025-10-24 17:00", 120, 0, "30m", "Education", 87, 1),
    ("Review Biology concepts", "2025-10-25 22:00", 90, 1, "none", "Education", 84, 1),
    ("Online course final project", "2025-10-26 20:00", 240, 0, "1d", "Education", 94, 1),
    ("Practice programming exercises", "2025-10-27 19:00", 120, 1, "none", "Education", 86, 1),
    ("Prepare for class debate", "2025-10-28 18:00", 90, 0, "1h", "Education", 82, 1),
    ("Study flashcards", "2025-10-29 21:00", 60, 1, "none", "Education", 78, 1),
    ("Complete homework assignments", "2025-10-30 20:00", 180, 0, "2h", "Education", 91, 1),
    ("Review lecture recordings", "2025-10-31 19:00", 120, 1, "none", "Education", 83, 1),
    
    # ========== PERSONAL TASKS (Mostly Neglected) ==========
    ("Grocery shopping", "2025-10-17 17:00", 60, 1, "2h", "Personal", 35, 0),  # Ordered delivery
    ("Laundry", "2025-10-18 19:00", 90, 1, "none", "Personal", 28, 0),  # Piled up for weeks
    ("Call family", "2025-10-19 21:00", 30, 1, "1h", "Personal", 45, 0),  # Too busy studying
    ("Clean dorm room", "2025-10-20 16:00", 45, 1, "none", "Personal", 32, 0),  # Room is messy
    ("Cook proper meal", "2025-10-21 18:30", 60, 1, "1h", "Personal", 38, 0),  # Ate instant noodles
    ("Gym workout", "2025-10-22 07:00", 60, 1, "none", "Personal", 42, 0),  # Studied instead
    ("Social event", "2025-10-23 20:00", 120, 0, "1d", "Personal", 55, 0),  # Cancelled to study
    ("Budget planning", "2025-10-24 19:00", 30, 1, "none", "Personal", 48, 0),  # Forgot completely
    ("Movie with friends", "2025-10-25 20:00", 180, 0, "2h", "Personal", 52, 0),  # Said no
    ("Self-care day", "2025-10-26 14:00", 120, 1, "none", "Personal", 40, 0),  # No time
    
    # ========== WORK TASKS (Bare Minimum) ==========
    ("Library shift", "2025-10-17 13:00", 240, 0, "30m", "Work", 75, 1),  # Only for money
    ("Tutoring session", "2025-10-18 15:00", 90, 0, "1h", "Work", 82, 0),  # Cancelled for study
    ("Campus job", "2025-10-19 10:00", 180, 0, "1d", "Work", 78, 1),  # Bare minimum
    ("Research assistant", "2025-10-20 14:00", 120, 0, "2h", "Work", 85, 0),  # Skipped
    ("Coffee shop shift", "2025-10-21 16:00", 240, 0, "1h", "Work", 72, 1),  # Need the money
    ("Freelance project", "2025-10-22 19:00", 90, 1, "1d", "Work", 79, 0),  # Delayed
    ("TA office hours", "2025-10-23 13:00", 120, 0, "30m", "Work", 76, 1),  # Required
    ("Internship work", "2025-10-24 09:00", 180, 0, "1d", "Work", 83, 0)   # Prioritized studying
]
    
    # Clear existing tasks
    c.execute("DELETE FROM tasks")
    
    # Insert new tasks
    c.executemany("""
        INSERT INTO tasks (title, deadline, duration, is_flexible, reminders, category, score, isCompleted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tasks)
    
    conn.commit()
    
    # Verify insertion
    c.execute("SELECT COUNT(*) FROM tasks")
    count = c.fetchone()[0]
    
    conn.close()
    
    print(f"✅ Seeded {count} tasks into the database!")
    return count

if __name__ == "__main__":
    init_database()
    seed_tasks()