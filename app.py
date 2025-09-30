from flask import Flask, render_template, request, redirect, url_for

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
    task = {
        'title': request.form.get('title'),
        'date': request.form.get('date'),
        'time': request.form.get('time'),
        'priority': request.form.get('priority'),
        'category': request.form.get('category')
    }
    tasks.append(task)
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    app.run(debug=True)

