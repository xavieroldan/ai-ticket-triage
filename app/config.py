"""Load and validate the label configuration used by the classifier."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "labels.yaml"


class ConfigError(ValueError):
    """Raised when labels.yaml is missing required data or is malformed."""


@dataclass(frozen=True)
class LabelConfig:
    categories: dict
    urgency: dict
    review_threshold: float
    hypothesis_template: str

    def category_hypotheses(self) -> dict:
        """Map each category key to the NLI hypothesis text."""
        return {k: self.hypothesis_template.format(v) for k, v in self.categories.items()}

    def urgency_hypotheses(self) -> dict:
        """Map each urgency key to the NLI hypothesis text."""
        return {k: self.hypothesis_template.format(v) for k, v in self.urgency.items()}


def _label_map(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict) or len(value) < 2:
        raise ConfigError(f"'{key}' must define at least two labels")
    for name, description in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"'{key}' contains an empty label name")
        if not isinstance(description, str) or not description.strip():
            raise ConfigError(f"label '{name}' in '{key}' needs a description")
    return dict(value)


def parse_config(data: dict) -> LabelConfig:
    """Validate a parsed YAML document and build a LabelConfig."""
    if not isinstance(data, dict):
        raise ConfigError("labels file must contain a mapping")

    threshold = data.get("review_threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ConfigError("'review_threshold' must be a number between 0 and 1")

    template = data.get("hypothesis_template", "This support ticket is about: {}.")
    if not isinstance(template, str) or template.count("{}") != 1:
        raise ConfigError("'hypothesis_template' must contain exactly one '{}' placeholder")

    return LabelConfig(
        categories=_label_map(data, "categories"),
        urgency=_label_map(data, "urgency"),
        review_threshold=float(threshold),
        hypothesis_template=template,
    )


def load_config(path=DEFAULT_PATH) -> LabelConfig:
    """Read labels.yaml from disk and validate it."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"labels file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return parse_config(yaml.safe_load(fh))
