import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import TextPreprocessor
from src.feature_engineering import FeatureExtractor

X_TRAIN = np.array([
    "This movie was absolutely wonderful, a real delight.",
    "Terrible film, a complete waste of time.",
    "Best movie I have seen in years, brilliant acting.",
    "Awful, boring, and far too long. Do not watch.",
    "A joy from start to finish, wonderful performances.",
    "I hated this movie, it was not good at all.",
])
Y_TRAIN = np.array([1, 0, 1, 0, 1, 0])


def _build_fitted_pipeline():
    preprocessor = TextPreprocessor()
    fe = FeatureExtractor(strategy="tfidf_word", max_features=500)

    X_clean = preprocessor.fit_transform(X_TRAIN)
    X_vec = fe.fit_transform(X_clean)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, Y_TRAIN)

    return Pipeline([
        ("preprocess", preprocessor),
        ("vectorize", fe),
        ("clf", clf),
    ])


def test_pipeline_predicts_one_label_per_input():
    pipeline = _build_fitted_pipeline()
    preds = pipeline.predict(["A new review to classify."])
    assert len(preds) == 1
    assert preds[0] in (0, 1)


def test_list_and_dataframe_input_give_identical_predictions():
    pipeline = _build_fitted_pipeline()
    texts = ["Wonderful film, loved every minute.", "Dull and disappointing."]

    preds_from_list = pipeline.predict(texts)
    preds_from_dataframe = pipeline.predict(pd.DataFrame({"text": texts}))

    assert list(preds_from_list) == list(preds_from_dataframe)


def test_predict_proba_is_a_valid_probability_distribution():
    pipeline = _build_fitted_pipeline()
    proba = pipeline.predict_proba(pd.DataFrame({"text": ["Fantastic movie."]}))
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-6
