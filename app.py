from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

FIELDS = [
    ("MedInc", "Median income (in $10,000s)"),
    ("HouseAge", "House age (years)"),
    ("AveRooms", "Average rooms"),
    ("AveBedrms", "Average bedrooms"),
    ("Population", "Population of area"),
    ("AveOccup", "Average occupants"),
    ("Latitude", "Latitude"),
    ("Longitude", "Longitude"),
]

@app.route("/")
def home():
    return render_template("index.html", fields=FIELDS, result=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[name]) for name, _ in FIELDS]
        prediction = model.predict(np.array([values]))[0]
        result = f"Predicted house price: ${prediction * 100000:,.0f}"
        return render_template("index.html", fields=FIELDS, result=result, error=None)
    except Exception:
        return render_template("index.html", fields=FIELDS, result=None,
                               error="Please enter valid numbers in every box.")

if __name__ == "__main__":
    app.run(debug=True)