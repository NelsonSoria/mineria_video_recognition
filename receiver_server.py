# receiver_server.py
from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)
CSV_FILE = 'tracking_data.csv'

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['camera_id', 'person_id', 'x', 'y', 'timestamp'])

@app.route('/track', methods=['POST'])
def track():
    data = request.get_json()
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            data['camera_id'],
            data['person_id'],
            data['x'],
            data['y'],
            data['timestamp']
        ])
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
