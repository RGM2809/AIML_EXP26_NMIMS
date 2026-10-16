
from flask import Flask, request, jsonify
from flask import send_from_directory
import joblib
import numpy as np

app = Flask(__name__, static_folder="frontend", static_url_path="")

model = joblib.load("iris_model.joblib")
target_names = ["setosa", "versicolor", "virginica"]


# Display the website
@app.route("/", methods=["GET"])
def home():
    return send_from_directory("frontend", "index.html")


# Check whether the API is running
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Healthy",
        "message": "Iris AI API is running!"
    })


# Predict the Iris flower species
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"], dtype=float)

        if features.shape != (4,) or not np.all(np.isfinite(features)):
            return jsonify({
                "error": "Enter exactly four valid numeric measurements."
            }), 400

        if np.any(features < 0):
            return jsonify({
                "error": "Measurements cannot be negative."
            }), 400

        prediction = model.predict(features.reshape(1, -1))[0]

        return jsonify({
            "status": "Success",
            "input_features": features.tolist(),
            "prediction_code": int(prediction),
            "predicted_class": target_names[int(prediction)]
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Send four numeric values under 'features'."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
