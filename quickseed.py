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
    # ========== WORK TASKS (Balanced completion) ==========
    ("Morning work planning", "2025-10-17 09:00", 60, 0, "30m", "Work", 75, 1),
    ("Client meeting preparation", "2025-10-17 14:00", 90, 0, "1h", "Work", 82, 1),
    ("Team collaboration session", "2025-10-18 10:00", 120, 0, "1d", "Work", 78, 1),
    ("Project deadline", "2025-10-19 17:00", 180, 0, "1d,2h", "Work", 88, 1),
    ("Weekly report submission", "2025-10-20 16:00", 45, 1, "3h", "Work", 70, 0),  # Will do tomorrow
    ("Skill development workshop", "2025-10-21 13:00", 120, 0, "1d", "Work", 80, 1),
    ("Work emails organization", "2025-10-22 08:00", 60, 1, "none", "Work", 65, 1),
    ("Networking event", "2025-10-23 18:00", 90, 0, "1d", "Work", 72, 1),
    
    # ========== EDUCATION TASKS (Consistent learning) ==========
    ("Online course module", "2025-10-17 20:00", 60, 1, "1h", "Education", 68, 1),
    ("Read industry book", "2025-10-18 21:00", 45, 1, "none", "Education", 62, 1),
    ("Research new technology", "2025-10-19 19:00", 90, 1, "1d", "Education", 75, 1),
    ("Practice new skill", "2025-10-20 20:30", 60, 1, "30m", "Education", 70, 0),  # Skipped for rest
    ("Watch educational video", "2025-10-21 21:00", 30, 1, "none", "Education", 58, 1),
    ("Study group participation", "2025-10-22 19:00", 120, 0, "1d", "Education", 78, 1),
    ("Learning journal update", "2025-10-23 20:00", 30, 1, "none", "Education", 65, 1),
    
    # ========== PERSONAL TASKS (Good self-care) ==========
    ("Morning workout", "2025-10-17 07:00", 60, 1, "none", "Personal", 55, 1),
    ("Grocery shopping", "2025-10-18 17:00", 45, 1, "2h", "Personal", 48, 1),
    ("Cook healthy dinner", "2025-10-19 18:30", 60, 1, "1h", "Personal", 52, 1),
    ("Call family", "2025-10-20 20:00", 30, 1, "1h", "Personal", 58, 1),
    ("Meditation session", "2025-10-21 07:30", 20, 1, "none", "Personal", 45, 0),  # Overslept
    ("Home cleaning", "2025-10-22 16:00", 90, 1, "none", "Personal", 50, 1),
    ("Social gathering", "2025-10-23 19:00", 120, 0, "1d", "Personal", 62, 1),
    ("Personal finance review", "2025-10-24 18:00", 60, 1, "1d", "Personal", 68, 1),
    ("Hobby time (guitar)", "2025-10-25 20:00", 45, 1, "none", "Personal", 42, 1),
    ("Weekend hiking trip", "2025-10-26 09:00", 240, 0, "1d", "Personal", 72, 1),
    ("Read fiction book", "2025-10-27 21:00", 60, 1, "none", "Personal", 48, 1),
    ("Self-reflection journal", "2025-10-28 22:00", 30, 1, "none", "Personal", 55, 0),  # Too tired
    
    # ========== EXTRA TASKS ==========
    ("Work project brainstorming", "2025-10-29 10:00", 90, 0, "30m", "Work", 79, 1),
    ("Learn new software tool", "2025-10-29 20:00", 60, 1, "none", "Education", 74, 1),
    ("Plan next vacation", "2025-10-30 19:00", 45, 1, "none", "Personal", 60, 1)
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