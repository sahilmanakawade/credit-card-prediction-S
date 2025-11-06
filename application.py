from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
grid_model = pickle.load(open("grid1.pkl", "rb"))
standard_scaler = pickle.load(open("z_score.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        try:
            features = [
                float(request.form['AGE']),
                float(request.form['EXPERIENCE']),
                float(request.form['INCOME']),
                float(request.form['ZIP CODE']),
                float(request.form['FAMILY']),
                float(request.form['CCAvg']),
                float(request.form['EDUCATION']),
                float(request.form['MORTGAGE']),
                float(request.form['PERSONAL LOAN']),
                float(request.form['SECURITIES ACCOUNT']),
                float(request.form['CD ACCOUNT']),
                float(request.form['ONLINE'])
            ]

            final_features = standard_scaler.transform([features])
            prediction = grid_model.predict(final_features)[0]

            results = "CREDIT CARD GRANTED" if prediction == 1 else "CAN'T GRANT CREDIT"

        except Exception:
            error = "Invalid input values. Please enter correct numerical data."

    return render_template("index.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)