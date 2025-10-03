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
    start_date = request.form.get("start_date")
    start_time = request.form.get("start_time")
    end_date = request.form.get("end_date")
    end_time = request.form.get("end_time")
    category = request.form.get('category')
    single_day = request.form.get("singleDay")

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
    


    if not end_date or end_date.lower() == "none":
        end_date = start_date

    if not start_date or not start_time:
        return "Start date and time are required.", 400
    
    try:
        start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "Invalid date or time format.", 400

    duration = end_dt - start_dt
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes = int(remainder // 60)
    duration_str = f"{int(hours)}h {minutes}m"



#CALCULATION OF PRIORITY SCORE
    #calculates the time remaining from time to task time
    #urgency = datetime.now() - start_dt
    # Ensure duration is defined
    duration = end_dt - start_dt
    duration_min = duration.total_seconds() / 60

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

# Urgency logic
    delta_min = (datetime.now() - start_dt).total_seconds() / 60
    urgency_custom = 1000 + delta_min if delta_min >= 0 else 1000 / (1 + abs(delta_min) / 60)

# Duration penalty scaled by importance
    duration_penalty = (duration_min * 0.1) / (1 + importance)

# Final score
    score = (urgency_custom * 0.35) + (importance * 0.7) - duration_penalty

    #might need this
    #reminder_times = [start_dt - offset for offset in valid_reminders]
    #lead_time_min = max(0, (start_dt - datetime.now()).total_seconds() / 60)


    #testing
    print("Start date:", start_date)
    print("Start time:", start_time)
    print("Parsed start_dt:", start_dt)
    print("Reminders received:", reminders)
    print("Valid reminders:", valid_reminders)
    print("Reminder count:", reminder_value_total)
    print("Category weight:", category_weight)
    print("Urgency:", urgency_custom)
    print("Duration (min):", duration_min)
    print("Importance:", importance)
    print("Score:", score)
   
    task = {
        "title": title,
        "start": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end": end_dt.strftime("%Y-%m-%d %H:%M"),
        "duration": duration_str,
        "reminder": has_reminder,
        "category": category
        }
    tasks.append(task)
    return redirect(url_for('dashboard'))
    
    

  
if __name__ == "__main__":
    app.run(debug=True)

