import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

plt.rcParams.update({
    "figure.dpi":        120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
})


class Evaluator:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_prob=None, model_name: str = "Model"):
        metrics = {
            "accuracy":        accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            except Exception:
                metrics["roc_auc"] = None

        bar = "=" * 55
        print(f"\n{bar}")
        print(f"  Evaluation Report — {model_name}")
        print(bar)
        print(f"  Accuracy          : {metrics['accuracy']:.4f}")
        print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro)    : {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro)        : {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted)     : {metrics['f1_weighted']:.4f}")
        if "roc_auc" in metrics and metrics["roc_auc"] is not None:
            print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
        print(bar)
        print("\n  Per-class Report:")
        print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str = "Model"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"confusion_matrix_{model_name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Confusion matrix saved → {path}")

    def plot_roc_curve(self, y_true, y_prob, model_name: str = "Model"):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc         = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, color="#2563EB", label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"roc_curve_{model_name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] ROC curve saved → {path}")

    def plot_top_features(self, feature_names, coefficients,
                          model_name: str = "Model", top_n: int = 20):
        if coefficients is None:
            print("[Evaluator] Skipping feature plot — model has no linear coefficients.")
            return

        feature_names = np.array(feature_names)
        coef          = np.array(coefficients)

        top_pos_idx = np.argsort(coef)[-top_n:][::-1]
        top_neg_idx = np.argsort(coef)[:top_n]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, indices, title, color in zip(
            axes,
            [top_pos_idx, top_neg_idx],
            [f"Top {top_n} Positive Features", f"Top {top_n} Negative Features"],
            ["#16A34A", "#DC2626"],
        ):
            vals   = coef[indices]
            labels = feature_names[indices]
            ax.barh(range(len(vals)), vals[::-1], color=color, alpha=0.85)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels[::-1], fontsize=8)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Coefficient", fontsize=10)

        fig.suptitle(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"feature_importance_{model_name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Feature importance plot saved → {path}")

    def compare_models(self, results: dict):
        model_names     = list(results.keys())
        metrics_to_plot = ["accuracy", "f1_macro", "roc_auc"]
        colors          = ["#2563EB", "#16A34A", "#D97706"]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))

        for ax, metric, color in zip(axes, metrics_to_plot, colors):
            values = [results[m].get(metric, 0) or 0 for m in model_names]
            bars   = ax.bar(model_names, values, color=color, alpha=0.85, width=0.5)
            ax.set_ylim(min(values) - 0.05, 1.02)
            ax.set_title(metric.replace("_", " ").upper(), fontsize=11, fontweight="bold")
            ax.set_ylabel("Score", fontsize=10)
            ax.tick_params(axis="x", rotation=20)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                )

        fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "model_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluator] Model comparison chart saved → {path}")