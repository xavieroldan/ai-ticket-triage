import pytest

from app.config import ConfigError, load_config, parse_config


def valid():
    return {
        "categories": {"billing": "Invoices", "bug": "Errors"},
        "urgency": {"high": "Blocking", "low": "Minor"},
        "review_threshold": 0.6,
        "hypothesis_template": "About: {}.",
    }


def test_repository_labels_file_is_valid():
    config = load_config()
    assert "billing" in config.categories
    assert set(config.urgency) == {"high", "medium", "low"}
    assert 0 <= config.review_threshold <= 1


def test_hypotheses_use_template():
    config = parse_config(valid())
    assert config.category_hypotheses() == {"billing": "About: Invoices.", "bug": "About: Errors."}
    assert config.urgency_hypotheses()["high"] == "About: Blocking."


def test_defaults_are_applied():
    data = valid()
    del data["review_threshold"]
    del data["hypothesis_template"]
    config = parse_config(data)
    assert config.review_threshold == 0.5
    assert "{}" in config.hypothesis_template


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(categories={"only": "one"}),
        lambda d: d.update(urgency="not a mapping"),
        lambda d: d["categories"].update(empty=""),
        lambda d: d.update(review_threshold=1.5),
        lambda d: d.update(hypothesis_template="no placeholder"),
    ],
)
def test_invalid_configs_are_rejected(mutate):
    data = valid()
    mutate(data)
    with pytest.raises(ConfigError):
        parse_config(data)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
