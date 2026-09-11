import os
import sys
from flask import Flask, request, jsonify, render_template
import pandas as pd
import cloudpickle

app = Flask(__name__)

BASE_DIR                 = os.path.dirname(os.path.abspath(__file__))
BAKED_MODEL_PATH         = os.path.join(BASE_DIR, "deploy", "model")
LEGACY_JOBLIB_PATH       = os.path.join(BASE_DIR, "outputs", "sentiment_pipeline_logistic_regression.joblib")
LABEL_MAP                = {0: "NEGATIVE", 1: "POSITIVE"}

_pipeline         = None
_legacy_predictor = None
_model_source     = None


def _load_from_local_export(model_dir):
    code_dir = os.path.join(model_dir, "code")
    if os.path.isdir(code_dir) and code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        return cloudpickle.load(f)


def _load_from_mlflow_uri(uri):
    import mlflow.sklearn
    return mlflow.sklearn.load_model(uri)


def _load_model():
    global _pipeline, _legacy_predictor, _model_source

    env_uri = os.environ.get("MODEL_URI")
    if env_uri:
        try:
            _pipeline = _load_from_mlflow_uri(env_uri)
            _model_source = "mlflow_uri"
            print(f"[App] Loaded model from MODEL_URI env var: {env_uri}")
            return
        except Exception as e:
            print(f"[App] Could not load model from MODEL_URI ({env_uri}): {e}")

    try:
        _pipeline = _load_from_local_export(BAKED_MODEL_PATH)
        _model_source = "direct"
        print(f"[App] Loaded model from baked-in champion export: {BAKED_MODEL_PATH}")
        return
    except Exception as e:
        print(f"[App] Could not load baked-in model export ({BAKED_MODEL_PATH}): {e}")

    print("[App] No model available — falling back to legacy joblib pipeline...")
    try:
        from src.predictor import SentimentPredictor
        _legacy_predictor = SentimentPredictor.load(LEGACY_JOBLIB_PATH)
        _model_source = "legacy_joblib"
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
    return jsonify({"status": "ok" if _model_source else "no_model_loaded", "model_source": _model_source})


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
