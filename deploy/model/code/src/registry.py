"""
MLflow Model Registry helpers — Phase 3.

Centralizes the "one registered model, many versions, aliases mark
lifecycle state" pattern so training (main.py) and, later, the serving
app (Phase 4) share the exact same logic for finding the current champion.

Uses aliases (`champion` / `challenger`) rather than the legacy Stages
API (Staging/Production/Archived), which MLflow deprecated in 2.9 and is
being removed in a future major release.
"""

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import mlflow.sklearn

REGISTERED_MODEL_NAME = "sentiment-analysis-classifier"
CHAMPION_ALIAS   = "champion"
CHALLENGER_ALIAS = "challenger"
PROMOTION_METRIC = "f1_macro"  # compared on the held-out TEST set


def _get_alias_version(client: MlflowClient, name: str, alias: str):
    """Return the ModelVersion currently behind `alias`, or None if the
    registered model or the alias doesn't exist yet (first-ever run)."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _metric_for_version(client: MlflowClient, version) -> float | None:
    run = client.get_run(version.run_id)
    return run.data.metrics.get(f"test_{PROMOTION_METRIC}")


def register_and_promote(model_info, test_metrics: dict,
                          registered_model_name: str = REGISTERED_MODEL_NAME):
    """
    Decides whether the version just registered (via
    mlflow.sklearn.log_model(..., registered_model_name=...)) should become
    the new '{CHAMPION_ALIAS}'.

    model_info      : the ModelInfo object returned by log_model()
    test_metrics    : the dict already computed by Evaluator.evaluate() on
                       the TEST split for this run — reused here so we don't
                       need a second round trip to MLflow for a value we
                       already have in memory.

    Rule: the new version is promoted if there is no current champion yet,
    or if it beats the current champion on PROMOTION_METRIC. Whichever
    version loses out (the old champion when replaced, or the new version
    when it doesn't win) is tagged '{CHALLENGER_ALIAS}', so the most recent
    runner-up is always visible for comparison.
    """
    client      = MlflowClient()
    new_version = model_info.registered_model_version
    new_score   = test_metrics.get(PROMOTION_METRIC)

    if new_version is None:
        print("[Registry] log_model() didn't return a registered version — "
              "was registered_model_name passed to log_model()?")
        return

    if new_score is None:
        print(f"[Registry] '{PROMOTION_METRIC}' not found in test metrics — skipping promotion check.")
        return

    champion = _get_alias_version(client, registered_model_name, CHAMPION_ALIAS)

    if champion is None:
        client.set_registered_model_alias(registered_model_name, CHAMPION_ALIAS, str(new_version))
        print(f"[Registry] No existing champion — v{new_version} promoted to "
              f"'{CHAMPION_ALIAS}' (test_{PROMOTION_METRIC}={new_score:.4f}).")
        return

    champion_score = _metric_for_version(client, champion)

    if champion_score is None:
        client.set_registered_model_alias(registered_model_name, CHALLENGER_ALIAS, str(new_version))
        print(f"[Registry] Could not read champion v{champion.version}'s test_{PROMOTION_METRIC} — "
              f"leaving champion as-is, tagged v{new_version} as '{CHALLENGER_ALIAS}'.")
        return

    print(f"[Registry] Champion v{champion.version}: test_{PROMOTION_METRIC}={champion_score:.4f}  "
          f"|  New v{new_version}: test_{PROMOTION_METRIC}={new_score:.4f}")

    if new_score > champion_score:
        client.set_registered_model_alias(registered_model_name, CHALLENGER_ALIAS, str(champion.version))
        client.set_registered_model_alias(registered_model_name, CHAMPION_ALIAS, str(new_version))
        print(f"[Registry] v{new_version} beats the champion — promoted to '{CHAMPION_ALIAS}'. "
              f"Previous champion (v{champion.version}) is now '{CHALLENGER_ALIAS}'.")
    else:
        client.set_registered_model_alias(registered_model_name, CHALLENGER_ALIAS, str(new_version))
        print(f"[Registry] v{new_version} did not beat the champion — tagged as '{CHALLENGER_ALIAS}'. "
              f"Champion remains v{champion.version}.")


def load_champion_model(registered_model_name: str = REGISTERED_MODEL_NAME):
    """Load the current champion pipeline. This is what Phase 4's serving
    app will call instead of loading a local .joblib file — the app always
    gets whatever the registry currently considers best, no redeploy needed."""
    uri = f"models:/{registered_model_name}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(uri)