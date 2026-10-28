"""New execution inputs cannot carry ignored or ambiguous fold settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from tradingdev.domain.validation.config import (
    AutomaticWalkForwardConfig,
    ExplicitWalkForwardConfig,
    PurgedWalkForwardConfig,
    parse_walk_forward_config,
)
from tradingdev.domain.validation.date_windows import WalkForwardSplitError
from tradingdev.domain.validation.splitter import DataSplitter

if TYPE_CHECKING:
    import pandas as pd

_EXPLICIT = {
    "mode": "explicit",
    "train_start": "2024-01-01",
    "train_end": "2024-01-02",
    "test_start": "2024-01-03",
    "test_end": "2024-01-04",
}


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ({"mode": "automatic"}, AutomaticWalkForwardConfig),
        (_EXPLICIT, ExplicitWalkForwardConfig),
        ({"mode": "purged", "label_horizon": 5}, PurgedWalkForwardConfig),
    ],
)
def test_modes_round_trip_without_irrelevant_defaults(
    raw: dict[str, Any],
    kind: type,
) -> None:
    config = parse_walk_forward_config(raw)
    assert isinstance(config, kind)
    saved = config.model_dump(mode="json")
    assert parse_walk_forward_config(saved) == config
    if config.mode != "explicit":
        assert "train_start" not in saved
    if config.mode != "purged":
        assert "embargo_bars" not in saved
    with pytest.raises(ValidationError, match="frozen"):
        cast("Any", config).target_metric = "total_return"


@pytest.mark.parametrize("raw", [{}, {"n_splits": 3}, {"mode": "unknown"}])
def test_mode_is_required_at_the_input_boundary(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="tag"):
        parse_walk_forward_config(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"mode": "automatic", "train_start": None},
        {"mode": "automatic", "embargo_bars": 0},
        {**_EXPLICIT, "n_splits": 1},
        {**_EXPLICIT, "train_ratio": 0.8},
        {**_EXPLICIT, "expanding": False},
        {"mode": "purged", "label_horizon": 1, "train_start": None},
        {"mode": "purged", "label_horizon": 1, "train_ratio": 0.8},
        {"mode": "purged", "label_horizon": 1, "expanding": True},
        {"mode": "purged", "label_horizon": 1, "purged_split": True},
    ],
)
def test_inactive_fields_are_errors_even_with_null_or_default_values(
    raw: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        parse_walk_forward_config(raw)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
@pytest.mark.parametrize("mode", ["automatic", "purged"])
def test_fold_counts_are_positive_integers(value: object, mode: str) -> None:
    with pytest.raises(ValidationError):
        parse_walk_forward_config({"mode": mode, "n_splits": value, "label_horizon": 1})


@pytest.mark.parametrize("value", [0, 1, -0.1, float("nan"), float("inf"), True, "0.5"])
def test_ratio_is_finite_and_strictly_inside_unit_interval(value: object) -> None:
    with pytest.raises(ValidationError):
        parse_walk_forward_config({"mode": "automatic", "train_ratio": value})


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
@pytest.mark.parametrize("field", ["label_horizon", "embargo_bars"])
def test_purge_controls_are_nonnegative_integers(value: object, field: str) -> None:
    with pytest.raises(ValidationError):
        parse_walk_forward_config({"mode": "purged", "label_horizon": 1, field: value})


@pytest.mark.parametrize(
    "raw", [{"mode": "purged"}, {"mode": "purged", "label_horizon": 0}]
)
def test_purged_requires_a_positive_horizon(raw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="label_horizon"):
        parse_walk_forward_config(raw)


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_automatic_folds_fail_clearly_when_data_cannot_form_windows(
    sample_ohlcv_df: pd.DataFrame,
    rows: int,
) -> None:
    splitter = DataSplitter(AutomaticWalkForwardConfig(mode="automatic", n_splits=3))
    with pytest.raises(WalkForwardSplitError, match="nonempty"):
        splitter.split(sample_ohlcv_df.iloc[:rows])
