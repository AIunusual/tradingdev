"""Adapt selected Maverick recommendation rules, without its server dependencies.

Derived from wshobson/maverick-mcp, commit
da70d074dcdac5fb21e3b2e6383cfebf391f28b5, backtesting/analysis.py
_generate_recommendations. MIT license is distributed beside this module.
Changes: evaluate rules independently, preserve missing values, use saved closed
trade metrics, omit Kelly/risk-reward rules absent from our metric contract, and
express recommendations as experiments instead of promised improvements/live use.
No upstream grade, investment suitability verdict or causal attribution is used.
"""

from dataclasses import dataclass
from typing import Literal

from tradingdev.domain.diagnostics import (
    DiagnosticContext,
    DiagnosticFinding,
    DiagnosticResult,
    MetricEvidence,
)

SOURCE = (
    "https://github.com/wshobson/maverick-mcp/blob/"
    "da70d074dcdac5fb21e3b2e6383cfebf391f28b5/"
    "maverick/backtesting/analysis.py"
)
VERSION = "da70d074-adapter-1"


@dataclass(frozen=True)
class _Rule:
    metric: str
    threshold: float
    operator: Literal["lt", "gt"]
    unit: str
    suggestion: str


_RULES = {
    "drawdown": _Rule(
        "max_drawdown",
        0.25,
        "gt",
        "fraction",
        "Inspect trades in drawdowns; compare an explicit exit/risk-control "
        "variant against the unchanged baseline before claiming improvement.",
    ),
    "win_rate": _Rule(
        "win_rate",
        0.45,
        "lt",
        "fraction",
        "Inspect losing entries and payoff sizes before proposing an entry filter; "
        "win rate alone does not establish profitability.",
    ),
    "few_trades": _Rule(
        "total_trades",
        20,
        "lt",
        "count",
        "Check coverage and signal conditions. A broader authorized sample or "
        "a sensitivity experiment can assess whether the evidence is sufficient.",
    ),
    "many_trades": _Rule(
        "total_trades",
        200,
        "gt",
        "count",
        "Check turnover and charged costs before testing a signal filter; "
        "an absolute trade count does not establish overtrading.",
    ),
    "profit_factor": _Rule(
        "profit_factor",
        1.2,
        "lt",
        "ratio",
        "Inspect loss sizes and exit behavior, then compare one explicit exit "
        "variant with the baseline on unchanged costs and evaluation windows.",
    ),
    "sharpe": _Rule(
        "sharpe_ratio",
        1.0,
        "lt",
        "ratio",
        "Inspect return variability and exposure before proposing a sizing "
        "experiment; volatility sensitivity has not been measured by this rule.",
    ),
}


class MaverickDiagnostics:
    """An interchangeable initial provider, not the product's research policy."""

    def diagnose(self, context: DiagnosticContext) -> DiagnosticResult:
        scope = context.performance
        findings: list[DiagnosticFinding] = []
        evaluated: list[str] = []
        unavailable = {
            "risk_reward": "not_in_saved_metric_contract",
            "kelly": "not_in_saved_metric_contract",
        }
        for code, rule in _RULES.items():
            value = scope.values.get(rule.metric)
            if scope.kind != "backtest":
                unavailable[code] = "aggregate_scope_requires_original_result"
            elif scope.mode != "signal" and code in {"drawdown", "sharpe"}:
                unavailable[code] = "requires_portfolio_return_mode"
            elif not isinstance(value, int | float) or isinstance(value, bool):
                unavailable[code] = "metric_unavailable"
            else:
                evaluated.append(code)
                triggered = (
                    value < rule.threshold
                    if rule.operator == "lt"
                    else value > rule.threshold
                )
                if triggered:
                    findings.append(
                        DiagnosticFinding(
                            code=code,
                            observation=(
                                f"Saved {rule.metric}={value:g} is "
                                f"{'below' if rule.operator == 'lt' else 'above'} "
                                f"this provider's review threshold "
                                f"{rule.threshold:g} ({rule.unit})."
                            ),
                            evidence=[
                                MetricEvidence(
                                    metric=rule.metric,
                                    value=float(value),
                                    unit=rule.unit,
                                )
                            ],
                            suggested_check=rule.suggestion,
                        )
                    )
        return DiagnosticResult(
            provider="maverick",
            version=VERSION,
            source=SOURCE,
            status="evaluated" if evaluated else "unavailable",
            findings=findings,
            evaluated_rules=evaluated,
            unavailable_rules=unavailable,
            limitations=[
                "Fixed heuristic thresholds, not user risk limits or proof of causes.",
                "Uses this scope's saved settings and closed-trade metrics; "
                "counts are not normalized by sample length.",
                "No market-regime, volatility-sensitivity or out-of-sample "
                "robustness diagnosis. No finding does not imply a safe/good strategy.",
                "Suggested experiments are untested and do not authorize execution.",
            ],
        )
