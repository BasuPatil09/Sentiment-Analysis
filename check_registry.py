from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.registry import REGISTERED_MODEL_NAME, PROMOTION_METRIC


def main():
    client = MlflowClient()
    print(f"Registered model: {REGISTERED_MODEL_NAME}\n")

    try:
        registered_model = client.get_registered_model(REGISTERED_MODEL_NAME)
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except MlflowException:
        print("No versions registered yet — train a model with main.py first.")
        return

    alias_by_version = {}
    for alias, version in registered_model.aliases.items():
        alias_by_version.setdefault(int(version), []).append(alias)

    rows = []
    for v in versions:
        run   = client.get_run(v.run_id)
        score = run.data.metrics.get(f"test_{PROMOTION_METRIC}")
        model = run.data.params.get("model", "?")
        rows.append((int(v.version), model, score, alias_by_version.get(int(v.version), [])))

    rows.sort(key=lambda r: r[0])

    print(f"{'Version':<8}{'Model':<22}{'test_' + PROMOTION_METRIC:<18}{'Alias'}")
    print("-" * 60)
    for version, model, score, aliases in rows:
        score_str = f"{score:.4f}" if score is not None else "n/a"
        alias_str = ", ".join(aliases) if aliases else ""
        print(f"{version:<8}{model:<22}{score_str:<18}{alias_str}")


if __name__ == "__main__":
    main()
