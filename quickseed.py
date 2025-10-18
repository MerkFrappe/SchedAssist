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
    
    tasks = [
           ("Urgent Client Deliverable", "2025-01-20 17:00", 120, 0, "1d,2h", "Work", 95, 1),
    ("Team Strategy Meeting", "2025-01-20 10:00", 90, 0, "30m", "Work", 85, 1),
    ("Project Deadline", "2025-01-21 23:59", 180, 0, "1d,3h", "Work", 98, 1),
    ("Client Presentation Prep", "2025-01-22 14:00", 120, 0, "1h", "Work", 88, 1),
    ("Budget Review", "2025-01-23 16:00", 60, 0, "2h", "Work", 82, 1),
    ("Weekly Report", "2025-01-24 12:00", 45, 0, "1h", "Work", 75, 1),
    ("System Update", "2025-01-25 09:00", 90, 0, "30m", "Work", 78, 0),  # Only 1 incomplete
    ("Code Deployment", "2025-01-26 15:00", 60, 0, "1h", "Work", 80, 1),

    # PERSONAL/HOME TASKS
     ("Clean Garage", "2025-01-20 15:00", 120, 1, "none", "Personal", 30, 0),
    ("Organize Files", "2025-01-21 16:00", 90, 1, "none", "Personal", 25, 0),
    ("Learn Guitar", "2025-01-22 19:00", 60, 1, "none", "Personal", 35, 1),  # Only 1 completed
    ("Meditation Practice", "2025-01-23 07:00", 30, 1, "none", "Personal", 20, 0),
    ("Journal Writing", "2025-01-24 21:00", 45, 1, "none", "Personal", 28, 0),
    ("Home DIY Project", "2025-01-25 14:00", 180, 1, "none", "Personal", 40, 0),

    # EDUCATION/SKILL DEVELOPMENT
    ("Learn New Framework Tutorial", "2025-01-21 15:00", 90, 1, "1h", "Education", 68, 0),
    ("Read Industry Articles", "2025-01-22 16:00", 45, 1, "none", "Education", 60, 0),
    ("Online Course - Advanced Skills", "2025-01-23 19:00", 120, 1, "1h", "Education", 75, 0),
    ("Practice Presentation Skills", "2025-01-25 16:30", 60, 1, "30m", "Education", 65, 0),

    # COMPLETED TASKS (for ML training)
    ("Update Work Portfolio", "2025-01-19 15:00", 90, 1, "1h", "Work", 72, 1),
    ("Research New Tools", "2025-01-19 11:00", 60, 1, "none", "Education", 58, 1),
    ("Plan Weekly Meals", "2025-01-19 18:00", 30, 1, "none", "Personal", 35, 1),
    ("Clear Email Inbox", "2025-01-19 10:00", 45, 1, "none", "Work", 62, 1)
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