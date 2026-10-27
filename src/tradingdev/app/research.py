"""Resolve explicit research choices before saving or capturing an execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from tradingdev.domain.backtest.schemas import BacktestRunConfig
from tradingdev.domain.data.crawlers.registry import describe_sources
from tradingdev.domain.performance.sampling import daily_observation_unavailable_reason
from tradingdev.domain.research import (
    InheritResearchSpec,
    NewResearchSpec,
    ResearchMetadata,
    ResearchRevisionIdentity,
    ResearchSource,
    ResearchSpec,
    research_datetime,
)
from tradingdev.domain.research_policy import (
    CRYPTO_DAILY_PERIODS_PER_YEAR,
    DEFAULT_FEES,
    DEFAULT_INIT_CASH,
    DEFAULT_SLIPPAGE,
    RESEARCH_POLICY_VERSION,
)

_CORE_FIELDS = ("symbol", "timeframe", "start_date", "end_date")
_COST_DEFAULTS = {
    "init_cash": DEFAULT_INIT_CASH,
    "fees": DEFAULT_FEES,
    "slippage": DEFAULT_SLIPPAGE,
}
_RESEARCH_BACKTEST_FIELDS = {*_CORE_FIELDS, *_COST_DEFAULTS, "periods_per_year"}


@dataclass(frozen=True)
class SelectedResearchRevision:
    """An immutable revision already selected and verified by the storage boundary."""

    strategy_id: str
    revision_id: str
    config: dict[str, Any]
    config_hash: str | None = None

    def identity(self) -> ResearchRevisionIdentity:
        return ResearchRevisionIdentity(
            strategy_id=self.strategy_id,
            revision_id=self.revision_id,
            config_hash=self.config_hash,
        )


def build_research_config(
    raw_config: dict[str, Any],
    research: ResearchSpec,
    *,
    selected_revision: SelectedResearchRevision | None = None,
) -> dict[str, Any]:
    """Build complete authored YAML; duplicate owned fields must agree with intake."""
    return _resolve(
        raw_config,
        research,
        selected_revision=selected_revision,
        backtest_overrides=None,
        check_authored_fields=True,
    )


def resolve_research_config(
    raw_config: dict[str, Any],
    research: ResearchSpec,
    *,
    selected_revision: SelectedResearchRevision | None = None,
    backtest_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace template research values and preserve the exact resolved provenance."""
    return _resolve(
        raw_config,
        research,
        selected_revision=selected_revision,
        backtest_overrides=backtest_overrides,
        check_authored_fields=False,
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _data_settings(
    config: dict[str, Any], selected: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = _mapping(config.get("data", {}), "data")
    supplied_data = dict(data)
    if selected is not None:
        data = {**_mapping(selected.get("data", {}), "selected data"), **data}
    requirements = _mapping(data.get("requirements", {}), "data.requirements")
    market = _mapping(requirements.get("market", {}), "data.requirements.market")
    supplied_requirements = _mapping(
        supplied_data.get("requirements", {}), "data.requirements"
    )
    supplied_market = _mapping(
        supplied_requirements.get("market", {}), "data.requirements.market"
    )
    if "source" in supplied_data and "source" not in supplied_market:
        market["source"] = supplied_data["source"]
    source = market.get("source", data.get("source"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "Research needs explicit data.source or data.requirements.market.source; "
            "choose the provider instead of inheriting the default Binance source"
        )
    descriptors = {item["source"]: item for item in describe_sources()}
    descriptor = descriptors.get(source)
    if descriptor is None:
        raise ValueError(f"Unknown research data source {source!r}")
    market_types = descriptor["market_types"]
    if market_types and data.get("market_type") not in market_types:
        raise ValueError(
            f"Research source {source!r} requires explicit data.market_type; "
            f"choose one of {', '.join(market_types)}"
        )
    market["source"] = source
    requirements["market"] = market
    data["requirements"] = requirements
    data["source"] = source
    return data, market, descriptor


def _market_identity(config: dict[str, Any]) -> tuple[object, object, object]:
    backtest = _mapping(config.get("backtest", {}), "backtest")
    data = _mapping(config.get("data", {}), "data")
    requirements = _mapping(data.get("requirements", {}), "data.requirements")
    market = _mapping(requirements.get("market", {}), "data.requirements.market")
    return (
        backtest.get("symbol"),
        market.get("source", data.get("source")),
        data.get("market_type"),
    )


def _annualization(
    research: ResearchSpec,
    *,
    current: dict[str, Any],
    selected: dict[str, Any] | None,
    descriptor: dict[str, Any],
) -> tuple[float | None, ResearchSource, str]:
    choice = research.annualization
    if (
        isinstance(research, InheritResearchSpec)
        and "annualization" not in research.model_fields_set
        and selected is not None
        and _market_identity(current) == _market_identity(selected)
    ):
        backtest = _mapping(selected.get("backtest", {}), "selected backtest")
        if "periods_per_year" in backtest:
            return (
                backtest["periods_per_year"],
                "selected_revision",
                "selected_revision",
            )
    if choice is None:
        choice = "market"
    if choice == "unavailable":
        return None, "request", "explicitly_unavailable"
    if isinstance(choice, int | float):
        return float(choice), "request", "explicit_daily_factor"
    if set(descriptor["asset_classes"]) == {"crypto"}:
        return (
            CRYPTO_DAILY_PERIODS_PER_YEAR,
            "project_default",
            "crypto_365_observed_utc_days",
        )
    raise ValueError(
        f"No market annualization policy applies to {descriptor['source']!r}. "
        "Provide a positive annualization factor for observed UTC daily returns "
        "or explicitly choose annualization='unavailable'; mixed providers do not "
        "identify an exchange calendar."
    )


def _equivalent(field: str, left: object, right: object) -> bool:
    if field in {"start_date", "end_date"}:
        return research_datetime(left) == research_datetime(right)
    # True == 1 and NaN coercions must not turn invalid input into agreement.
    if isinstance(left, bool):
        return False
    return bool(left == right)


def _resolve(
    raw_config: dict[str, Any],
    research: ResearchSpec,
    *,
    selected_revision: SelectedResearchRevision | None,
    backtest_overrides: dict[str, Any] | None,
    check_authored_fields: bool,
) -> dict[str, Any]:
    research = TypeAdapter(ResearchSpec).validate_python(research)
    inherited = isinstance(research, InheritResearchSpec)
    if inherited and selected_revision is None:
        raise ValueError(
            "inherit_selected_revision requires a selected immutable revision; "
            "bundled examples cannot supply missing research choices"
        )
    selected = selected_revision.config if inherited and selected_revision else None
    selected_backtest = (
        _mapping(selected.get("backtest", {}), "selected backtest")
        if selected is not None
        else {}
    )
    selected_research = (
        _mapping(selected.get("research", {}), "selected research")
        if selected is not None
        else {}
    )
    config = deepcopy(raw_config)
    backtest = _mapping(config.get("backtest", {}), "backtest")
    authored_backtest = dict(backtest)
    sources: dict[str, ResearchSource] = {}
    outline = research.strategy_outline
    if outline is None:
        outline = selected_research.get("strategy_outline")
        sources["/research/strategy_outline"] = "selected_revision"
    else:
        sources["/research/strategy_outline"] = "request"
    if not isinstance(outline, str) or not outline.strip():
        raise ValueError(
            "The selected revision has no strategy_outline; provide it explicitly"
        )
    for name in _CORE_FIELDS:
        value = getattr(research, name)
        source: ResearchSource = "request"
        if value is None:
            value = selected_backtest.get(name)
            source = "selected_revision"
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(
                f"The selected revision has no {name}; provide it explicitly"
            )
        if name in {"start_date", "end_date"}:
            value = research_datetime(value)
        backtest[name] = value
        sources[f"/backtest/{name}"] = source
    for name, default in _COST_DEFAULTS.items():
        value = getattr(research, name)
        source = "request"
        if name not in research.model_fields_set:
            if name in selected_backtest:
                value, source = selected_backtest[name], "selected_revision"
            else:
                value, source = default, "project_default"
                if name == "init_cash" and backtest.get("mode") == "volume":
                    value = None
        backtest[name] = value
        sources[f"/backtest/{name}"] = source
    config["backtest"] = backtest
    data, market, descriptor = _data_settings(config, selected)
    authored_data = _mapping(raw_config.get("data", {}), "data")
    authored_requirements = _mapping(
        authored_data.get("requirements", {}), "data.requirements"
    )
    authored_market = _mapping(
        authored_requirements.get("market", {}), "data.requirements.market"
    )
    for name in ("symbol", "timeframe"):
        market[name] = backtest[name]
    config["data"] = data
    factor, source, reason = _annualization(
        research, current=config, selected=selected, descriptor=descriptor
    )
    backtest["periods_per_year"] = factor
    sources["/backtest/periods_per_year"] = source
    overrides = dict(backtest_overrides or {})
    for name, value in overrides.items():
        if name in _RESEARCH_BACKTEST_FIELDS:
            raise ValueError(
                f"backtest_overrides.{name} conflicts with research ownership; "
                "put research choices in ResearchSpec"
            )
        else:
            backtest[name] = value
    if check_authored_fields:
        for name in _RESEARCH_BACKTEST_FIELDS.intersection(authored_backtest):
            if not _equivalent(name, authored_backtest[name], backtest[name]):
                raise ValueError(f"YAML backtest.{name} conflicts with research")
        for name in ("symbol", "timeframe"):
            if name in authored_market and authored_market[name] != backtest[name]:
                raise ValueError(
                    f"YAML data.requirements.market.{name} conflicts with research"
                )
        authored_research = _mapping(config.get("research", {}), "research")
        if (
            "strategy_outline" in authored_research
            and authored_research["strategy_outline"] != outline
        ):
            raise ValueError("YAML research.strategy_outline conflicts with research")
    sampling_reason = daily_observation_unavailable_reason(backtest["timeframe"])
    if sampling_reason is not None:
        reason = f"{reason}; {sampling_reason}"
    # Inherited YAML is evidence, not a reason to coerce invalid/null selections.
    effective_inputs = {
        "kind": "new",
        "strategy_outline": outline,
        **{name: backtest[name] for name in _CORE_FIELDS},
        "fees": backtest["fees"],
        "slippage": backtest["slippage"],
        "annualization": factor if factor is not None else "unavailable",
    }
    if backtest["init_cash"] is not None:
        effective_inputs["init_cash"] = backtest["init_cash"]
    NewResearchSpec.model_validate(effective_inputs)
    metadata = ResearchMetadata(
        policy_version=RESEARCH_POLICY_VERSION,
        strategy_outline=outline,
        field_sources=sources,
        selected_revision=(
            selected_revision.identity() if inherited and selected_revision else None
        ),
        annualization_reason=reason,
    )
    config["research"] = metadata.model_dump(mode="json")
    return BacktestRunConfig.model_validate(config).model_dump(mode="json")
