import os
import joblib
import numpy as np

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}
EMOJI_MAP  = {0: "👎", 1: "👍"}


class SentimentPredictor:
    def __init__(self, preprocessor=None, feature_extractor=None, trainer=None):
        self.preprocessor      = preprocessor
        self.feature_extractor = feature_extractor
        self.trainer           = trainer

    def save(self, path: str):
        payload = {
            "preprocessor":      self.preprocessor,
            "feature_extractor": self.feature_extractor,
            "trainer":           self.trainer,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(payload, path)
        print(f"[Predictor] Pipeline saved → {path}")

    @classmethod
    def load(cls, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved pipeline at: {path}")
        payload  = joblib.load(path)
        instance = cls(
            preprocessor      = payload["preprocessor"],
            feature_extractor = payload["feature_extractor"],
            trainer           = payload["trainer"],
        )
        print(f"[Predictor] Pipeline loaded ← {path}")
        return instance

    def predict_one(self, text: str, verbose: bool = True) -> dict:
        result = self.predict_batch([text], verbose=False)[0]
        if verbose:
            emoji = EMOJI_MAP[1 if result["label"] == "POSITIVE" else 0]
            print(f"\n  Text       : \"{result['text'][:80]}\"")
            print(f"  Sentiment  : {emoji}  {result['label']}")
            if result["confidence"] is not None:
                print(f"  Confidence : {result['confidence']:.2%}")
            else:
                print(f"  Confidence : N/A (model does not support probability scores)")
        return result

    def predict_batch(self, texts, verbose: bool = True) -> list:
        clean = self.preprocessor.transform(texts)
        X     = self.feature_extractor.transform(clean)
        preds = self.trainer.predict(X)
        try:
            probs     = self.trainer.predict_proba(X)
            has_proba = True
        except AttributeError:
            probs     = np.array([0.5] * len(preds))
            has_proba = False
            print("[Predictor] Note: this model does not support probabilities. Confidence scores unavailable.")

        results = []
        for text, pred, prob in zip(texts, preds, probs):
            label = LABEL_MAP[pred]
            conf  = (prob if pred == 1 else (1 - prob)) if has_proba else None
            results.append({
                "text":       text,
                "label":      label,
                "confidence": float(conf) if conf is not None else None,
            })

        if verbose:
            print(f"\n[Predictor] Predicted {len(results)} samples.")
        return results

    def interactive_demo(self):
        print("\n" + "=" * 55)
        print("  Sentiment Analysis — Interactive Demo")
        print("  Type a movie/product review. ('exit' to quit)")
        print("=" * 55)
        while True:
            try:
                text = input("\n  Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting demo.")
                break
            if text.lower() in ("exit", "quit", "q"):
                print("  Exiting demo.")
                break
            if not text:
                continue
            self.predict_one(text, verbose=True)