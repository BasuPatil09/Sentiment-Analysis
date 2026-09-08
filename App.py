import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import mlflow.sklearn

app = Flask(__name__)

BASE_DIR                 = os.path.dirname(os.path.abspath(__file__))
BAKED_MODEL_PATH         = os.path.join(BASE_DIR, "deploy", "model")
LEGACY_JOBLIB_PATH       = os.path.join(BASE_DIR, "outputs", "sentiment_pipeline_logistic_regression.joblib")
LABEL_MAP                = {0: "NEGATIVE", 1: "POSITIVE"}

_pipeline         = None
_legacy_predictor = None


def _load_model():
    global _pipeline, _legacy_predictor

    candidates = []
    env_uri = os.environ.get("MODEL_URI")
    if env_uri:
        candidates.append(("MODEL_URI env var", env_uri))
    candidates.append(("baked-in champion export", BAKED_MODEL_PATH))

    for source, uri in candidates:
        try:
            _pipeline = mlflow.sklearn.load_model(uri)
            print(f"[App] Loaded model from {source}: {uri}")
            return
        except Exception as e:
            print(f"[App] Could not load model from {source} ({uri}): {e}")

    print("[App] No MLflow model available — falling back to legacy joblib pipeline...")
    try:
        from src.predictor import SentimentPredictor
        _legacy_predictor = SentimentPredictor.load(LEGACY_JOBLIB_PATH)
    except Exception as e:
        print(f"[App] Legacy fallback also failed: {e}")
        print("[App] WARNING: no model loaded. /predict will return an error until one is available.")


_load_model()


def predict_text(text: str) -> dict:
    if _pipeline is not None:
        df    = pd.DataFrame({"text": [text]})
        pred  = int(_pipeline.predict(df)[0])
        label = LABEL_MAP[pred]
        confidence = None
        if hasattr(_pipeline, "predict_proba"):
            proba = _pipeline.predict_proba(df)[0]
            confidence = float(proba[pred])
        return {"text": text, "label": label, "confidence": confidence}

    if _legacy_predictor is not None:
        return _legacy_predictor.predict_one(text, verbose=False)

    raise RuntimeError("No model is loaded.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    if _pipeline is not None:
        source = "mlflow"
    elif _legacy_predictor is not None:
        source = "legacy_joblib"
    else:
        source = None
    return jsonify({"status": "ok" if source else "no_model_loaded", "model_source": source})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text"}), 400
    try:
        result = predict_text(text)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    return jsonify(result)


@app.route("/compare")
def compare():
   models = {
    "Logistic Regression": {"accuracy": 0.9024, "f1_macro": 0.9024, "roc_auc": 0.9635},
    "Linear SVM":          {"accuracy": 0.9040, "f1_macro": 0.9040, "roc_auc": 0.9652},
    "Naive Bayes":         {"accuracy": 0.8840, "f1_macro": 0.8840, "roc_auc": 0.9533},
    "Random Forest":       {"accuracy": 0.8572, "f1_macro": 0.8572, "roc_auc": 0.9356},
    "SGD Classifier":      {"accuracy": 0.9060, "f1_macro": 0.9060, "roc_auc": 0.9660},
    }
   return jsonify(models)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
