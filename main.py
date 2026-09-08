import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from src.data_loader         import load_csv, split_data
from src.preprocessor        import TextPreprocessor
from src.feature_engineering import FeatureExtractor
from src.models               import ModelTrainer, MODEL_REGISTRY
from src.evaluator            import Evaluator
from src.predictor            import SentimentPredictor
from src.registry             import register_and_promote, REGISTERED_MODEL_NAME


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Analysis — Training Pipeline")
    p.add_argument("--csv",       type=str, default=None)
    p.add_argument("--text_col",  type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--model",     type=str, default="logistic_regression",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--strategy",  type=str, default="tfidf_word",
                   choices=["tfidf_word", "tfidf_char", "tfidf_combo", "bow"])
    p.add_argument("--tune",      action="store_true")
    p.add_argument("--compare",   action="store_true")
    p.add_argument("--output",    type=str, default="outputs")
    return p.parse_args()


def _get_coefficients(trainer):
    model = trainer.get_model()

    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        coef = getattr(inner, "coef_", None)
        return coef.ravel() if coef is not None else np.zeros(1)

    coef = getattr(model, "coef_", None)
    return coef.ravel() if coef is not None else None


def _log_existing_artifact(path):
    if os.path.exists(path):
        mlflow.log_artifact(path)


def _log_full_pipeline_as_model(preprocessor, fe, trainer, X_train, model_name):
    full_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("vectorize",  fe),
        ("clf",        trainer.get_model()),
    ])

    sample_input  = pd.DataFrame({"text": list(X_train[:3])})
    sample_output = full_pipeline.predict(sample_input)
    signature     = infer_signature(sample_input, sample_output)

    try:
        model_info = mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            name="model",
            signature=signature,
            input_example=sample_input,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            code_paths=["src"],
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)
        print(f"\n[MLflow] Full pipeline logged as a Model → {model_info.model_uri}")
        print(
            "[MLflow] Load it anywhere with: "
            f"mlflow.sklearn.load_model('{model_info.model_uri}')"
        )
        return model_info

    except Exception as e:
        print(f"[MLflow] Model logging skipped due to an error: {e}")
        return None


def run_pipeline(args):

    os.makedirs(args.output, exist_ok=True)

    mlflow.set_experiment("sentiment-analysis")

    if args.csv:
        df = load_csv(
            args.csv,
            text_col=args.text_col,
            label_col=args.label_col
        )
    else:
        raise ValueError("Please provide a CSV file using --csv.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    run_name = args.model

    if args.tune:
        run_name += "_tuned"

    if args.compare:
        run_name += "_comparison"

    with mlflow.start_run(run_name=run_name):

        print(f"\n[MLflow] Run: {run_name}")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

        mlflow.log_param("model", args.model)
        mlflow.log_param("feature_strategy", args.strategy)
        mlflow.log_param("text_column", args.text_col)
        mlflow.log_param("label_column", args.label_col)
        mlflow.log_param("dataset_size", len(df))

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("validation_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))

        mlflow.log_param("tfidf_max_features", 50_000)

        mlflow.log_param("tuning_enabled", args.tune)
        mlflow.log_param("comparison_enabled", args.compare)

        print("\n[Pipeline] Preprocessing text...")

        preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True
        )

        X_train_clean = preprocessor.fit_transform(X_train)
        X_val_clean   = preprocessor.transform(X_val)
        X_test_clean  = preprocessor.transform(X_test)

        print("\n[Pipeline] Extracting features...")

        fe = FeatureExtractor(
            strategy=args.strategy,
            max_features=50_000
        )

        X_train_vec = fe.fit_transform(X_train_clean)
        X_val_vec   = fe.transform(X_val_clean)
        X_test_vec  = fe.transform(X_test_clean)

        mlflow.log_param(
            "num_features",
            X_train_vec.shape[1]
        )

        evaluator = Evaluator(output_dir=args.output)


        if args.compare:

            print("\n[Pipeline] Benchmarking all classifiers...")

            all_results = {}

            for name in MODEL_REGISTRY:

                print(f"\n{'─' * 45}")

                t = ModelTrainer(model_name=name)

                t.train(X_train_vec, y_train)

                preds = t.predict(X_val_vec)

                try:
                    probs = t.predict_proba(X_val_vec)
                except AttributeError:
                    probs = None

                metrics = evaluator.evaluate(
                    y_val,
                    preds,
                    probs,
                    model_name=name
                )

                all_results[name] = metrics

            evaluator.compare_models(all_results)

            print("\n[Pipeline] Model comparison complete.")

            for model_name, metrics in all_results.items():

                for metric_name, metric_value in metrics.items():

                    if isinstance(metric_value, (int, float, np.number)):
                        mlflow.log_metric(
                            f"{model_name}_{metric_name}",
                            float(metric_value)
                        )

            comparison_path = os.path.join(
                args.output,
                "model_comparison.png"
            )

            _log_existing_artifact(comparison_path)


        print(
            f"\n[Pipeline] Training primary model: '{args.model}'"
        )

        trainer = ModelTrainer(model_name=args.model)


        if args.tune:

            trainer.hyperparameter_tune(
                X_train_vec,
                y_train,
                cv=5
            )

        else:

            trainer.train(
                X_train_vec,
                y_train
            )

            trainer.cross_validate(
                X_train_vec,
                y_train,
                cv=5
            )


        print("\n[Pipeline] Evaluating on VALIDATION set...")

        val_preds = trainer.predict(X_val_vec)

        try:
            val_probs = trainer.predict_proba(X_val_vec)
        except AttributeError:
            val_probs = None

        val_metrics = evaluator.evaluate(
            y_val,
            val_preds,
            val_probs,
            model_name=f"{args.model}_val"
        )


        for metric_name, metric_value in val_metrics.items():

            if isinstance(metric_value, (int, float, np.number)):
                mlflow.log_metric(
                    f"val_{metric_name}",
                    float(metric_value)
                )


        print("\n[Pipeline] Evaluating on HELD-OUT TEST set...")

        test_preds = trainer.predict(X_test_vec)

        try:
            test_probs = trainer.predict_proba(X_test_vec)
        except AttributeError:
            test_probs = None

        final_metrics = evaluator.evaluate(
            y_test,
            test_preds,
            test_probs,
            model_name=f"{args.model}_test"
        )


        for metric_name, metric_value in final_metrics.items():

            if isinstance(metric_value, (int, float, np.number)):
                mlflow.log_metric(
                    f"test_{metric_name}",
                    float(metric_value)
                )


        if trainer.cv_mean is not None:
            mlflow.log_metric("cv_f1_mean", trainer.cv_mean)
            mlflow.log_metric("cv_f1_std", trainer.cv_std)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("cv_scoring", trainer.cv_scoring)

        if trainer.train_time is not None:
            mlflow.log_metric("train_time_seconds", trainer.train_time)

        if trainer.best_cv_score is not None:
            mlflow.log_metric("best_cv_f1", trainer.best_cv_score)

        if trainer.tuning_time is not None:
            mlflow.log_metric("tuning_time_seconds", trainer.tuning_time)

        if trainer.best_params is not None:
            for name, value in trainer.best_params.items():
                mlflow.log_param(f"best_{name}", value)


        evaluator.plot_confusion_matrix(
            y_test,
            test_preds,
            model_name=args.model
        )

        if test_probs is not None:

            evaluator.plot_roc_curve(
                y_test,
                test_probs,
                model_name=args.model
            )

        coef = _get_coefficients(trainer)

        if coef is not None:

            try:

                feature_names = fe.get_feature_names()

                evaluator.plot_top_features(
                    feature_names,
                    coef,
                    model_name=args.model,
                    top_n=20
                )

            except Exception as e:

                print(
                    f"[Pipeline] Feature importance plot skipped: {e}"
                )


        _log_existing_artifact(
            os.path.join(
                args.output,
                f"confusion_matrix_{args.model}.png"
            )
        )

        _log_existing_artifact(
            os.path.join(
                args.output,
                f"roc_curve_{args.model}.png"
            )
        )

        _log_existing_artifact(
            os.path.join(
                args.output,
                f"feature_importance_{args.model}.png"
            )
        )


        save_path = os.path.join(
            args.output,
            f"sentiment_pipeline_{args.model}.joblib"
        )

        predictor = SentimentPredictor(
            preprocessor=preprocessor,
            feature_extractor=fe,
            trainer=trainer
        )

        predictor.save(save_path)


        _log_existing_artifact(save_path)

        print("\n[Pipeline] Logging full pipeline as an MLflow Model (Phase 2)...")

        model_info = _log_full_pipeline_as_model(
            preprocessor,
            fe,
            trainer,
            X_train,
            model_name=args.model,
        )

        if model_info is not None:
            print(f"\n[Pipeline] Checking registry lifecycle (Phase 3)...")
            register_and_promote(model_info, final_metrics)


        print("\n[Pipeline] Quick inference smoke-test:")

        sample_texts = [
            "This movie was an absolute masterpiece. The acting was superb!",
            "Terrible film. Total waste of time. Boring and predictable.",
            "Not bad, had some good moments but also dragged in parts.",
        ]

        for r in predictor.predict_batch(
            sample_texts,
            verbose=False
        ):

            emoji = (
                "👍"
                if r["label"] == "POSITIVE"
                else "👎"
            )

            conf_str = (
                f"{r['confidence']:.1%}"
                if r["confidence"] is not None
                else "N/A"
            )

            print(
                f"  {emoji} [{conf_str}] "
                f"{r['text'][:65]}..."
            )

        print(
            "\n[Pipeline] All done! Outputs saved to:",
            args.output
        )

        print(
            "[MLflow] Run completed successfully."
        )

        return final_metrics


if __name__ == "__main__":

    args = parse_args()

    run_pipeline(args)
