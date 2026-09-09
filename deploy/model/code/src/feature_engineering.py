from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureExtractor(BaseEstimator, TransformerMixin):
    STRATEGIES = ["tfidf_word", "tfidf_char", "tfidf_combo", "bow"]

    def __init__(self, strategy: str = "tfidf_word", max_features: int = 50_000):
        self.strategy = strategy
        self.max_features = max_features

    def fit(self, X, y=None):
        self.vectorizer_ = self._build_vectorizer()
        print(f"[FeatureExtractor] Fitting '{self.strategy}' on {len(X)} samples...")
        self.vectorizer_.fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.vectorizer_ = self._build_vectorizer()
        print(f"[FeatureExtractor] Fitting '{self.strategy}' on {len(X)} samples...")
        Xt = self.vectorizer_.fit_transform(X)
        print(f"[FeatureExtractor] Feature matrix shape: {Xt.shape}")
        return Xt

    def transform(self, X):
        return self.vectorizer_.transform(X)

    def get_feature_names(self):
        return self.vectorizer_.get_feature_names_out()

    def _build_vectorizer(self):
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        if self.strategy == "tfidf_word":
            return TfidfVectorizer(
                analyzer      = "word",
                ngram_range   = (1, 2),
                max_features  = self.max_features,
                min_df        = 2,
                max_df        = 0.95,
                sublinear_tf  = True,
                strip_accents = "unicode",
            )

        elif self.strategy == "tfidf_char":
            return TfidfVectorizer(
                analyzer     = "char_wb",
                ngram_range  = (3, 5),
                max_features = self.max_features,
                min_df       = 3,
                sublinear_tf = True,
            )

        elif self.strategy == "tfidf_combo":
            word_vec = TfidfVectorizer(
                analyzer     = "word",
                ngram_range  = (1, 2),
                max_features = self.max_features // 2,
                min_df       = 2,
                max_df       = 0.95,
                sublinear_tf = True,
            )
            char_vec = TfidfVectorizer(
                analyzer     = "char_wb",
                ngram_range  = (3, 5),
                max_features = self.max_features // 2,
                min_df       = 3,
                sublinear_tf = True,
            )
            return FeatureUnion([("word", word_vec), ("char", char_vec)])

        elif self.strategy == "bow":
            return CountVectorizer(
                analyzer     = "word",
                ngram_range  = (1, 2),
                max_features = self.max_features,
                min_df       = 2,
                max_df       = 0.95,
            )
