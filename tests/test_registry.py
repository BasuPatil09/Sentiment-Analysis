import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.registry import (
    CHAMPION_ALIAS,
    CHALLENGER_ALIAS,
    register_and_promote,
)

TEST_MODEL_NAME = "test-sentiment-classifier"


@pytest.fixture
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test-experiment")
    return uri


def _log_fake_version(test_f1_macro: float):
    clf = LogisticRegression()
    clf.fit([[0], [1]], [0, 1])
    sample = pd.DataFrame({"x": [0]})

    with mlflow.start_run():
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            registered_model_name=TEST_MODEL_NAME,
            input_example=sample,
        )
    return model_info, {"f1_macro": test_f1_macro}


def test_first_version_becomes_champion_with_no_prior_history(tracking_uri):
    model_info, metrics = _log_fake_version(test_f1_macro=0.80)
    register_and_promote(model_info, metrics, registered_model_name=TEST_MODEL_NAME)

    client = MlflowClient()
    champion = client.get_model_version_by_alias(TEST_MODEL_NAME, CHAMPION_ALIAS)
    assert int(champion.version) == model_info.registered_model_version


def test_a_better_version_replaces_the_champion(tracking_uri):
    first_info, first_metrics = _log_fake_version(test_f1_macro=0.80)
    register_and_promote(first_info, first_metrics, registered_model_name=TEST_MODEL_NAME)

    second_info, second_metrics = _log_fake_version(test_f1_macro=0.90)
    register_and_promote(second_info, second_metrics, registered_model_name=TEST_MODEL_NAME)

    client = MlflowClient()
    champion = client.get_model_version_by_alias(TEST_MODEL_NAME, CHAMPION_ALIAS)
    challenger = client.get_model_version_by_alias(TEST_MODEL_NAME, CHALLENGER_ALIAS)

    assert int(champion.version) == second_info.registered_model_version
    assert int(challenger.version) == first_info.registered_model_version


def test_a_worse_version_does_not_replace_the_champion(tracking_uri):
    first_info, first_metrics = _log_fake_version(test_f1_macro=0.90)
    register_and_promote(first_info, first_metrics, registered_model_name=TEST_MODEL_NAME)

    second_info, second_metrics = _log_fake_version(test_f1_macro=0.80)
    register_and_promote(second_info, second_metrics, registered_model_name=TEST_MODEL_NAME)

    client = MlflowClient()
    champion = client.get_model_version_by_alias(TEST_MODEL_NAME, CHAMPION_ALIAS)
    challenger = client.get_model_version_by_alias(TEST_MODEL_NAME, CHALLENGER_ALIAS)

    assert int(champion.version) == first_info.registered_model_version
    assert int(challenger.version) == second_info.registered_model_version
