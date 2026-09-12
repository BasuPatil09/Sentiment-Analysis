import pandas as pd
from sklearn.base import clone

from src.preprocessor import TextPreprocessor


def test_get_params_returns_constructor_args_unchanged():
    tp = TextPreprocessor(remove_stopwords=False, lemmatize=False)
    assert tp.get_params() == {"remove_stopwords": False, "lemmatize": False}


def test_clone_does_not_raise():
    tp = TextPreprocessor()
    clone(tp)


def test_fit_transform_returns_one_string_per_input():
    tp = TextPreprocessor()
    out = tp.fit_transform(["A great movie.", "A terrible movie."])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(t, str) for t in out)


def test_html_tags_are_removed():
    tp = TextPreprocessor()
    out = tp.fit_transform(["<br/>This movie was <b>great</b>.<br/>"])
    assert "<" not in out[0]
    assert ">" not in out[0]


def test_negation_words_survive_stopword_removal():
    tp = TextPreprocessor(remove_stopwords=True)
    out = tp.fit_transform(["This movie was not good at all."])
    assert "not" in out[0].split()


def test_accepts_list_series_and_single_column_dataframe_identically():
    tp = TextPreprocessor()
    texts = ["A wonderful film.", "A dreadful film."]

    from_list = tp.fit_transform(texts)
    from_series = tp.transform(pd.Series(texts))
    from_dataframe = tp.transform(pd.DataFrame({"text": texts}))

    assert from_list == from_series
    assert from_list == from_dataframe


def test_dataframe_with_multiple_columns_raises():
    tp = TextPreprocessor()
    tp.fit(["warm up"])
    try:
        tp.transform(pd.DataFrame({"a": ["x"], "b": ["y"]}))
        assert False, "expected a ValueError for a multi-column DataFrame"
    except ValueError:
        pass
