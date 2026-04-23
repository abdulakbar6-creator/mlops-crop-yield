import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return jsonify({"status": "Crop Yield Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data])
    prediction = model.predict(features)[0]
    return jsonify({
        "predicted_yield_tons_per_hectare": round(float(prediction), 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
