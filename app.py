import json                                              
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)                                    

model = joblib.load("model.pkl")                         
with open("model_info.json") as f:
    info = json.load(f)

FEATURES = info["features"]                              
THRESHOLD = info["threshold"]
RANGES = info["ranges"]


@app.route("/")                                          
def home():
    return render_template("index.html")


@app.route("/api/info")                                  
def api_info():
    return jsonify(info)


@app.route("/api/predict", methods=["POST"])            
def api_predict():
    data = request.get_json()                           
    try:
        values = {name: float(data[name]) for name in FEATURES}   
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Missing or invalid values"}), 400

    row = pd.DataFrame([values])                         
    p_benign = float(model.predict_proba(row)[0][1])    
    label = "Benign" if p_benign >= THRESHOLD else "Malignant"   
    out_of_range = [                                
        n for n, v in values.items()
        if v < RANGES[n][0] or v > RANGES[n][1]
    ]

    return jsonify({                                    
        "label": label,
        "p_benign": round(p_benign * 100, 1),
        "p_malignant": round((1 - p_benign) * 100, 1),
        "threshold": THRESHOLD,
        "out_of_range": out_of_range,
    })


if __name__ == "__main__":                               
    app.run(debug=True)