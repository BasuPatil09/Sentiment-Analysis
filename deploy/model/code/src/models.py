import time
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV


MODEL_REGISTRY = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "linear_svm": {
        "model": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42)),
        "params": {
            "estimator__C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "naive_bayes": {
        "model": ComplementNB(),
        "params": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
        },
    },
    "sgd_classifier": {
        "model": SGDClassifier(
            loss="modified_huber",
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "alpha": [1e-4, 1e-3, 1e-2],
            "penalty": ["l2", "elasticnet"],
        },
    },
}


class ModelTrainer:
    def __init__(self, model_name: str = "logistic_regression"):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self._config = MODEL_REGISTRY[model_name]
        self.model = self._config["model"]
        self.best_params = None
        self.best_cv_score = None
        self.cv_mean = None
        self.cv_std = None
        self.cv_scoring = None
        self.tuning_time = None
        self.train_time = None

    def train(self, X_train, y_train):
        print(f"\n[ModelTrainer] Training '{self.model_name}'...")

        t0 = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - t0

        print(
            f"[ModelTrainer] Training complete in "
            f"{self.train_time:.2f}s"
        )

        return self

    def cross_validate(self, X, y, cv: int = 5, scoring: str = "f1"):
        print(
            f"\n[ModelTrainer] Running {cv}-fold CV "
            f"with scoring='{scoring}'..."
        )

        scores = cross_val_score(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        self.cv_mean = float(scores.mean())
        self.cv_std = float(scores.std())
        self.cv_scoring = scoring

        print(
            f"[ModelTrainer] CV {scoring.upper()}: "
            f"{self.cv_mean:.4f} ± {self.cv_std:.4f}"
        )

        return scores

    def hyperparameter_tune(
        self,
        X_train,
        y_train,
        cv: int = 5,
        scoring: str = "f1"
    ):
        param_grid = self._config["params"]

        print(
            f"\n[ModelTrainer] Hyperparameter tuning "
            f"'{self.model_name}' with {cv}-fold CV..."
        )

        gs = GridSearchCV(
            estimator=self._config["model"],
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )

        t0 = time.time()
        gs.fit(X_train, y_train)
        self.tuning_time = time.time() - t0

        self.model = gs.best_estimator_
        self.best_params = gs.best_params_
        self.best_cv_score = float(gs.best_score_)
        self.cv_scoring = scoring

        print(f"[ModelTrainer] Best params : {self.best_params}")
        print(
            f"[ModelTrainer] Best CV {scoring}: "
            f"{self.best_cv_score:.4f}"
        )
        print(
            f"[ModelTrainer] Tuning time : "
            f"{self.tuning_time:.2f}s"
        )

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]

        raise AttributeError(
            f"Model '{self.model_name}' does not support predict_proba."
        )

    def get_model(self):
        return self.model