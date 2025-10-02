from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

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
    reminders = request.form.getlist("reminders[]")
    has_reminder = any(r != "none" for r in reminders)

    title = request.form.get('title')
    start_date = request.form.get("start_date")
    start_time = request.form.get("start_time")
    end_date = request.form.get("end_date")
    end_time = request.form.get("end_time")
    category = request.form.get('category')

    start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")

    duration = end_dt - start_dt
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes = remainder // 60
    duration_str = f"{int(hours)}h {int(minutes)}m"

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

