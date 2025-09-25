from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("Main.html")  # loads templates/index.html

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/Main")
def schedule():
    return render_template("Main.html")

@app.route("/Analytics")
def analytics():
    return render_template("Analytics.html")

if __name__ == "__main__":
    app.run(debug=True)
