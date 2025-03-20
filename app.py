import os
import json
from flask import Flask, jsonify

app = Flask(__name__)

LATEST_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latest_prediction.json')

@app.route('/latest', methods=['GET'])
def get_latest():
    if not os.path.exists(LATEST_JSON):
        return jsonify({"error": "No predictions yet"}), 404
    
    with open(LATEST_JSON, 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    # Run on localhost:5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)