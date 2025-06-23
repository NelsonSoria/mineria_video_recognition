from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

CSV_FILE = "tracking_data.csv"

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/data")
def data():
    if not os.path.exists(CSV_FILE):
        return jsonify({"data": []})

    df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
    df = df.tail(100)  # últimos 100 registros
    records = df.to_dict(orient="records")
    return jsonify({"data": records})


if __name__ == "__main__":
    app.run(debug=True)
