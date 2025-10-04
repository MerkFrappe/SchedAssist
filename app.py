from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime, timedelta

app = Flask(__name__)

tasks = []  # Temporary in-memory list


@app.route("/")
def home():
    return render_template("Get_Started.html")

@app.route("/dashboard")
def dashboard():
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
    #deadline
    try:
        deadline = datetime.strptime(request.form.get('deadline'), "%Y-%m-%dT%H:%M")
    except ValueError:
        return "Invalid date or time format.", 400
    
    now = datetime.now()
    remaining_time = (deadline - now).total_seconds() / 60

    if remaining_time <= 0:
        urgency_score = 100  # overdue or immediate
    else:
        urgency_score = min((1 / remaining_time) * 100 + duration_minutes * 0.5, 100)
    
    
#CALCULATION OF PRIORITY SCORE
    #calculates the time remaining from time to task time
    #urgency = datetime.now() - start_dt
    # Ensure duration is defined

# Normalize category
    category = category.capitalize()

    match category:
        case "Personal":
            category_weight = 1
        case "Education" | "Work":
            category_weight = 2
        case _:
            category_weight = 1  # fallback instead of return

# Reminder parsing

    valid_reminders = [reminder_offsets[r] for r in reminders if r in reminder_offsets]
    reminder_value_total = sum(val for _, val in valid_reminders)

    importance = category_weight * reminder_value_total

# Flexibility 
    task_type = request.form.get('task_type', 'uninterrupted')
    is_flexible = task_type == 'flexible'

    flexibility_weight = 1.0 if is_flexible else 1.2


# Final score
    score = (urgency_score * 0.35) + (importance * 0.7) * flexibility_weight


    #testing
    print("Title:", title)
    print("Duration:", duration_minutes)
    print("Deadline", deadline)
    print("Reminders received:", reminders)
    print("Valid reminders:", valid_reminders)
    print("Reminder count:", reminder_value_total)
    print("Category weight:", category_weight)
    print("Flexible:", flexibility_weight)
    print("Importance:", importance)
    print("Score:", score)
   
    task = {
         "title": title,
        "deadline": deadline.strftime("%Y-%m-%d %H:%M"),
         "duration": f"{duration_minutes} min",
        "is_flexible": is_flexible,
        "reminders": reminders,
        "reminder": has_reminder,
        "category": category,
        }
    tasks.append(task)
    return redirect(url_for('dashboard'))
    
    

  
if __name__ == "__main__":
    app.run(debug=True)

