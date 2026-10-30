"""The declarative rolling schemes: names, overrides, and window arithmetic.

Every expected window bound below is arithmetic on the declared numbers, worked
out by hand. None of them are read back from the code under test, so a change to
the geometry has to disagree with the sum written here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingdev.domain.ml.rolling_gen import (
    ROLLING_SCHEMES,
    RollingGenConfig,
    RollingScheme,
    RollingSchemeConfig,
    RollType,
    get_rolling_scheme,
    resolve_geometry,
    rolling_scheme_names,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)


def test_built_in_scheme_names_are_stable() -> None:
    """The enumerable menu, in the order a caller may rely on."""
    assert rolling_scheme_names() == ("roll_sd", "roll_ex")


def test_every_name_resolves_to_itself() -> None:
    for name in rolling_scheme_names():
        assert get_rolling_scheme(name).name == name
    assert set(ROLLING_SCHEMES) == set(rolling_scheme_names())


def test_unknown_scheme_lists_the_alternatives() -> None:
    with pytest.raises(ValueError, match="unknown rolling scheme: rolling"):
        get_rolling_scheme("rolling")
    with pytest.raises(ValueError, match="roll_sd, roll_ex"):
        get_rolling_scheme("rolling")


def test_built_in_roll_types_match_their_names() -> None:
    assert get_rolling_scheme("roll_sd").rtype is RollType.ROLL_SD
    assert get_rolling_scheme("roll_ex").rtype is RollType.ROLL_EX


def test_overrides_replace_only_the_named_fields() -> None:
    scheme = RollingSchemeConfig(scheme="roll_sd", step=7).resolve()

    assert scheme.step == 7
    assert scheme.rtype is RollType.ROLL_SD
    # Untouched fields stay unset, so they keep following the strategy.
    assert scheme.window_size is None
    assert scheme.min_train_bars is None
    assert scheme.predict_lookback is None


def test_expanding_scheme_rejects_a_fixed_window_size() -> None:
    """A length that contradicts the expanding window must not be ignored."""
    with pytest.raises(ValueError, match="window_size only applies to ROLL_SD"):
        RollingSchemeConfig(scheme="roll_ex", window_size=240).resolve()


def test_misspelled_key_is_rejected_instead_of_ignored() -> None:
    """A YAML block with a typo must fail, not keep the built-in default."""
    with pytest.raises(ValidationError):
        RollingSchemeConfig.model_validate({"scheme": "roll_sd", "window_sise": 240})


def test_date_anchoring_is_not_accepted_yet() -> None:
    """``train_start`` is out of scope; declaring it fails instead of no-op."""
    with pytest.raises(ValidationError):
        RollingSchemeConfig.model_validate(
            {"scheme": "roll_sd", "train_start": "2023-01-01"}
        )


def test_the_window_shape_is_not_overridable() -> None:
    """The name decides the shape, so a contradicting ``rtype`` is refused.

    Allowing it would let ``scheme: roll_sd`` run an expanding window, which is
    exactly the confusion the scheme name exists to prevent.
    """
    with pytest.raises(ValidationError):
        RollingSchemeConfig.model_validate({"scheme": "roll_sd", "rtype": "ROLL_EX"})


def test_the_switch_defaults_to_off() -> None:
    assert RollingGenConfig().enabled is False


@pytest.mark.parametrize(
    "strategy_config", [XGBoostStrategyConfig, SafetyVolumeStrategyConfig]
)
@pytest.mark.parametrize(
    "declaration",
    [{"scheme": "unknown"}, {"scheme": "roll_ex", "window_size": 240}],
)
def test_invalid_enabled_scheme_fails_before_constructing_a_strategy(
    strategy_config: type[XGBoostStrategyConfig] | type[SafetyVolumeStrategyConfig],
    declaration: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        strategy_config.model_validate(
            {"rolling_gen": {"enabled": True, **declaration}}
        )


@pytest.mark.parametrize(
    "config",
    [XGBoostStrategyConfig(), SafetyVolumeStrategyConfig()],
    ids=["xgboost", "safety-volume"],
)
def test_strategies_are_off_by_default(
    config: XGBoostStrategyConfig | SafetyVolumeStrategyConfig,
) -> None:
    assert config.rolling_gen.enabled is False
    assert config.rolling_gen.scheme == "roll_sd"


def test_strategy_config_accepts_a_flat_rolling_gen_block() -> None:
    config = XGBoostStrategyConfig.model_validate(
        {"rolling_gen": {"enabled": True, "scheme": "roll_ex", "step": 48}}
    )

    assert config.rolling_gen.enabled is True
    assert config.rolling_gen.resolve().rtype is RollType.ROLL_EX
    assert config.rolling_gen.resolve().step == 48


def test_sliding_window_moves_with_the_bar() -> None:
    """lookback=10 declares a 10 x lookback = 100-bar window."""
    geometry = resolve_geometry(
        get_rolling_scheme("roll_sd"), lookback=10, retrain_interval=7
    )

    assert geometry.train_window(250) == (150, 250)
    # The window is clipped at the first bar, not pushed negative.
    assert geometry.train_window(40) == (0, 40)


def test_expanding_window_keeps_the_first_bar() -> None:
    geometry = resolve_geometry(
        get_rolling_scheme("roll_ex"), lookback=10, retrain_interval=7
    )

    assert geometry.train_window(250) == (0, 250)
    assert geometry.window_size is None


def test_declared_window_size_wins_over_the_lookback_rule() -> None:
    geometry = resolve_geometry(
        RollingScheme(
            name="roll_sd", description="", rtype=RollType.ROLL_SD, window_size=30
        ),
        lookback=10,
        retrain_interval=7,
    )

    assert geometry.train_window(250) == (220, 250)


@pytest.mark.parametrize("name", rolling_scheme_names())
def test_training_window_never_reaches_the_scored_bar(name: str) -> None:
    """The stop bound is the scored bar itself, for every roll type."""
    geometry = resolve_geometry(
        get_rolling_scheme(name), lookback=10, retrain_interval=7
    )

    for combined_idx in (0, 1, 99, 100, 250):
        _, stop = geometry.train_window(combined_idx)
        assert stop == combined_idx


def test_unset_fields_follow_the_strategy_parameters() -> None:
    """lookback=12, interval=50 -> window 120, min 22, context 24, step 50."""
    geometry = resolve_geometry(
        get_rolling_scheme("roll_sd"), lookback=12, retrain_interval=50
    )

    assert geometry.step == 50
    assert geometry.window_size == 120
    assert geometry.min_train_bars == 22
    assert geometry.predict_lookback == 24


def test_predict_context_ends_at_the_scored_bar() -> None:
    geometry = resolve_geometry(
        get_rolling_scheme("roll_sd"), lookback=10, retrain_interval=7
    )

    assert geometry.predict_context(250) == (230, 251)
    assert geometry.predict_context(3) == (0, 4)


def test_context_that_cannot_cover_the_lookback_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least lookback \\+ 1"):
        resolve_geometry(
            RollingScheme(
                name="roll_sd",
                description="",
                rtype=RollType.ROLL_SD,
                predict_lookback=10,
            ),
            lookback=10,
            retrain_interval=7,
        )


@pytest.mark.parametrize(
    ("scheme", "message"),
    [
        (RollingScheme("x", "", RollType.ROLL_SD, step=0), "step must be at least 1"),
        (
            RollingScheme("x", "", RollType.ROLL_SD, min_train_bars=0),
            "min_train_bars must be at least 1",
        ),
        (
            RollingScheme("x", "", RollType.ROLL_SD, window_size=0),
            "window_size must be at least 1",
        ),
    ],
    ids=["step", "min-train-bars", "window-size"],
)
def test_unusable_geometry_is_rejected(scheme: RollingScheme, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_geometry(scheme, lookback=10, retrain_interval=7)
