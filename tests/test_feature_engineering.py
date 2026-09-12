import pytest
from sklearn.base import clone

from src.feature_engineering import FeatureExtractor

SAMPLE_TEXTS = [
    "great movie loved it wonderful acting",
    "terrible film hated it awful acting",
    "amazing acting wonderful story great film",
    "boring plot awful acting terrible movie",
    "wonderful film great acting loved every part",
    "hated the boring terrible plot of this film",
]


def test_get_params_returns_constructor_args_unchanged():
    fe = FeatureExtractor(strategy="bow", max_features=100)
    assert fe.get_params() == {"strategy": "bow", "max_features": 100}


def test_clone_does_not_raise():
    fe = FeatureExtractor()
    clone(fe)


@pytest.mark.parametrize("strategy", FeatureExtractor.STRATEGIES)
def test_fit_transform_works_for_every_strategy(strategy):
    fe = FeatureExtractor(strategy=strategy, max_features=50)
    X = fe.fit_transform(SAMPLE_TEXTS)
    assert X.shape[0] == len(SAMPLE_TEXTS)
    assert X.shape[1] > 0


def test_transform_after_fit_has_matching_feature_count():
    fe = FeatureExtractor(strategy="tfidf_word", max_features=50)
    X_train = fe.fit_transform(SAMPLE_TEXTS)
    X_new = fe.transform(["a new review here"])
    assert X_new.shape[1] == X_train.shape[1]


def test_unknown_strategy_raises():
    fe = FeatureExtractor(strategy="not_a_real_strategy")
    with pytest.raises(ValueError):
        fe.fit(SAMPLE_TEXTS)
