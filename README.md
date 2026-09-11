# 🎬 Sentiment Analysis: NLP & MLOps Pipeline

A movie review sentiment classifier built with classical NLP and scikit-learn, trained on the IMDB dataset (50,000 reviews), and wrapped in a full MLOps lifecycle: experiment tracking, a model registry with automatic champion promotion, containerization, and a live deployment.

**Live demo:** [sentiment-analysis-icmn.onrender.com](https://sentiment-analysis-icmn.onrender.com)
*(free-tier hosting, so the first request after a period of inactivity can take 30-60 seconds while the instance wakes up)*

The current champion model (SGD Classifier) reaches **91.2% accuracy** and **0.97 ROC-AUC** on the held-out test set. Every training run is tracked, versioned, and compared automatically. See [Experiment Tracking & Model Registry](#experiment-tracking--model-registry) below for how the current best model is decided, not just claimed.

---

## 📸 Screenshots

| Model Comparison |
|---|
| ![Model Comparison](assets/model_comparison.png) |

| Confusion Matrix | ROC Curve | Feature Importance |
|---|---|---|
| ![CM](assets/confusion_matrix_logistic_regression.png) | ![ROC](assets/roc_curve_logistic_regression.png) | ![FI](assets/feature_importance_logistic_regression.png) |

---

## 🏆 Results

| Model | Accuracy | F1 (Macro) | ROC-AUC |
|---|---|---|---|
| SGD Classifier | **90.6%** | **0.906** | **0.966** |
| Linear SVM | 90.4% | 0.904 | 0.965 |
| Logistic Regression | 90.2% | 0.902 | 0.963 |
| Naive Bayes | 88.4% | 0.884 | 0.953 |
| Random Forest | 85.7% | 0.857 | 0.936 |

This table is a fixed benchmark from the initial model comparison (validation set). It doesn't update automatically; the registry does. Whichever model is currently serving in production is whatever `check_registry.py` reports as `champion`, decided by test-set F1 score each time a model is trained, not by editing this file.

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── src/
│   ├── __init__.py              # Package exports (kept minimal, see note below)
│   ├── data_loader.py           # CSV loading + train/val/test split
│   ├── preprocessor.py          # Text cleaning: sklearn-compatible transformer
│   ├── feature_engineering.py   # TF-IDF / BoW vectorizers: sklearn-compatible transformer
│   ├── models.py                # Model registry + GridSearchCV tuner
│   ├── evaluator.py             # Metrics, confusion matrix, ROC, feature plots
│   ├── predictor.py             # Local joblib-based inference (legacy path)
│   └── registry.py              # MLflow Model Registry: champion/challenger promotion
│
├── docker/
│   └── nltk_download.py         # Retry-safe NLTK corpus download for the image build
│
├── templates/
│   └── index.html               # Flask frontend
│
├── deploy/
│   └── model/                   # Exported champion model, generated, not hand-edited
│
├── assets/                      # Screenshots used in this README
├── data/                        # Dataset goes here (not committed)
├── outputs/                     # Saved plots + local joblib pipeline (not committed)
│
├── main.py                      # Trains, evaluates, logs to MLflow, registers, promotes
├── predict.py                   # Local inference CLI (joblib-based)
├── App.py                       # Flask app, serves whatever model was exported to deploy/model/
├── check_registry.py            # Prints every registered version, its score, and its alias
├── export_champion.py           # Exports a registered version into deploy/model/ for Docker
├── convert.py                   # IMDB CSV label converter
├── Dockerfile
├── .dockerignore
├── render.yaml                  # Render Blueprint
└── requirements.txt
```

`src/__init__.py` intentionally does not eagerly import every submodule. `evaluator.py` and `models.py` pull in matplotlib, seaborn, and scikit-learn's ensemble methods: useful for training, unnecessary weight for a serving container that only needs to unpickle a preprocessor and a vectorizer. `main.py` and `check_registry.py` import what they need directly from their specific modules, so nothing breaks; the serving path just stays lighter.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/BasuPatil09/Sentiment-Analysis.git
cd Sentiment-Analysis

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📦 Dataset

Download the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle and place it in the `data/` folder.

Then convert labels:
```bash
python convert.py
```
This maps `positive → 1` and `negative → 0` and saves `data/IMDB_clean.csv`.

---

## 🚀 Training

```bash
# Train with default Logistic Regression
python main.py --csv data/IMDB_clean.csv --text_col review --label_col sentiment

# Train a specific model
python main.py --csv data/IMDB_clean.csv --text_col review --label_col sentiment --model sgd_classifier

# Benchmark ALL models
python main.py --csv data/IMDB_clean.csv --text_col review --label_col sentiment --compare

# Hyperparameter tuning
python main.py --csv data/IMDB_clean.csv --text_col review --label_col sentiment --model logistic_regression --tune
```

Available models: `logistic_regression`, `linear_svm`, `naive_bayes`, `random_forest`, `sgd_classifier`

Available feature strategies: `tfidf_word`, `tfidf_char`, `tfidf_combo`, `bow`

Every run does more than print metrics to the console. See below.

---

## 🔬 Experiment Tracking & Model Registry

Each training run logs its parameters, metrics, and plots to MLflow, then goes further: it packages the fitted preprocessor, vectorizer, and classifier into one deployable pipeline, logs that as a versioned MLflow Model, and checks whether it beats the current best.

**View every run, including parameters, metrics, confusion matrices, and ROC curves:**
```bash
mlflow ui
```
Open `http://127.0.0.1:5000` and browse the `sentiment-analysis` experiment.

**Check which model version is currently serving:**
```bash
python check_registry.py
```
```
Version  Model                 test_f1_macro     Alias
------------------------------------------------------------
1        logistic_regression   0.9087            challenger
2        sgd_classifier        0.9124            champion
```

**How promotion works:** every registered version is compared against the current `champion` on held-out test F1. A new version that scores higher takes the `champion` alias; the model it replaced becomes `challenger`. A version that doesn't win is tagged `challenger` and the champion stays put. No manual step decides this. Training a model and letting it lose is the same action as training one and letting it win.

---

## 📦 Deployment

The `champion` model gets exported into a self-contained directory and built into a Docker image. No MLflow server needs to be reachable at serving time.

```bash
# Export whatever's currently champion
python export_champion.py

# Or export a specific version by number
python export_champion.py --version 3

# Build and run locally
docker build -t sentiment-analysis .
docker run -p 5000:5000 sentiment-analysis
```

```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This movie was absolutely wonderful."}'
```

`App.py` loads the exported model directly. No MLflow import is needed for the default path; only the `MODEL_URI` environment variable override pulls in MLflow, for setups with a reachable tracking server. The container runs on gunicorn with a single worker: a text pipeline that includes NLTK's WordNet corpus uses roughly 300-400 MB per process, which rules out multiple workers on a free-tier instance.

Deployed on [Render](https://render.com) via `render.yaml`. Connecting the repository as a Blueprint on Render's dashboard picks up the config automatically.

---

## 🔍 Local Inference (without the web app)

```bash
# Single prediction
python predict.py --model outputs/sentiment_pipeline_logistic_regression.joblib \
                  --text "The film was absolutely brilliant!"

# Batch from file (one review per line)
python predict.py --model outputs/sentiment_pipeline_logistic_regression.joblib \
                  --file data/new_reviews.txt

# Interactive CLI demo
python predict.py --model outputs/sentiment_pipeline_logistic_regression.joblib
```

This path uses the local joblib pipeline saved by `main.py`, independent of the MLflow registry. Useful for quick checks without touching Docker or the tracking server.

---

## 🌐 Web Application (local dev)

```bash
python App.py
```

Navigate to `http://127.0.0.1:5000`, type a review, and get a prediction with a confidence score alongside a live model comparison chart. This is the same app running at the [live demo link](https://sentiment-analysis-icmn.onrender.com), just without the container.

---

## 🧠 NLP Pipeline

```
Raw Text
  ↓ Lowercase + HTML removal
  ↓ Contraction expansion (can't → cannot)
  ↓ Special character removal
  ↓ Tokenization (NLTK punkt)
  ↓ Stopword removal (preserving negations: not, never, no)
  ↓ Lemmatization (WordNet)
  ↓ TF-IDF Vectorization (word unigrams + bigrams, 50k features)
  ↓ Classifier (Logistic Regression / SVM / NB / RF / SGD)
  ↓ Sentiment Label + Confidence Score
```

Key design decisions:
- **Negation words preserved** (`not`, `never`, `no`): dropping these as stopwords would erase exactly the words that flip a sentence's sentiment
- **Sublinear TF scaling** (`log(1+tf)`): keeps high-frequency terms from dominating the vector
- **ComplementNB** over MultinomialNB: performs better on the roughly balanced IMDB label split
- **CalibratedClassifierCV** wraps LinearSVC: LinearSVC has no native `predict_proba`, so this adds calibrated probability output
- **Preprocessor and vectorizer are real scikit-learn transformers** (`BaseEstimator`, `TransformerMixin`), not standalone helper classes: this is what lets the full pipeline, including text cleaning, get logged, versioned, and deployed as one MLflow Model instead of three separately-tracked pieces

---

## 📊 Visualizations Generated

After training, the following plots are saved to `outputs/` and logged as MLflow artifacts:

- `confusion_matrix_<model>.png`: predictions vs. ground truth
- `roc_curve_<model>.png`: ROC curve with AUC annotation
- `feature_importance_<model>.png`: top 20 positive & negative TF-IDF features
- `model_comparison.png`: bar chart comparing all models across metrics (`--compare` mode only)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| NLP | NLTK (tokenization, lemmatization, stopwords) |
| Feature Extraction | TF-IDF (word + char n-grams), Bag-of-Words |
| Experiment Tracking | MLflow (tracking, model logging, model registry) |
| Web Framework | Flask + gunicorn |
| Containerization | Docker |
| Deployment | Render |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Serialization | joblib, cloudpickle |
| Dataset | IMDB 50K Movie Reviews |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE)