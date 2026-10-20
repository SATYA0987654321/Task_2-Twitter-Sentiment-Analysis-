from flask import Flask, request, jsonify, send_from_directory
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "twitter_sentiment_model.pkl"

# Load the trained TF-IDF + LinearSVC pipeline
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "status": "ok",
            "message": "Twitter Sentiment API is running"
        })

    data = request.get_json(silent=True) or {}

    entity = data.get("entity", "")
    tweet = data.get("text", "")

    if not isinstance(entity, str) or not isinstance(tweet, str):
        return jsonify({"error": "Entity and tweet must be text"}), 400

    if not tweet.strip():
        return jsonify({"error": "Please enter a tweet"}), 400

    if len(tweet) > 5000 or len(entity) > 200:
        return jsonify({"error": "Input is too long"}), 400

    # Keep input format consistent with model training
    combined_text = entity.strip() + " " + tweet.strip()
    sentiment = model.predict([combined_text])[0]

    return jsonify({"sentiment": str(sentiment).title()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
