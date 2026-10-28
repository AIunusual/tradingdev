"""On-demand explanation workflow, separate from initial server instructions."""

from __future__ import annotations

from typing import Any

from tradingdev.domain.research_evidence import ResearchEvidenceDraft


def research_evidence_contract() -> dict[str, Any]:
    """Return binding rules and a schema-validated unsourced example."""
    example = ResearchEvidenceDraft.model_validate(
        {
            "presentation": {
                "title": "Long-only moving-average research",
                "summary": "Hold above the moving average; otherwise exit.",
                "parameter_descriptions": {
                    "/period": {
                        "label": "Moving-average lookback",
                        "description": "Completed closes used in the average.",
                        "unit": "candles",
                    }
                },
            },
            "rules": [
                {
                    "rule_id": "position",
                    "condition": "A complete, finite lookback is available.",
                    "action": "Hold long above the average; otherwise stay flat.",
                    "formula": (
                        "signal[t] = 1 if close[t] > SMA(close, period)[t] else 0"
                    ),
                    "variables": [
                        {
                            "symbol": "t",
                            "meaning": "Completed candle",
                            "unit": "candle",
                        },
                        {"symbol": "period", "meaning": "Lookback", "unit": "candles"},
                    ],
                }
            ],
            "limits": [
                "Illustrative model proposal; no external reference or return claim."
            ],
            "parameter_provenance": [
                {
                    "target_ids": ["base"],
                    "parameter_paths": ["/period"],
                    "provenance": "model_proposal",
                    "reason": "An explicitly unverified proposal for user review.",
                }
            ],
        }
    )
    return {
        "selection": [
            "Preserve explicit user choices. When parameters are omitted, propose a "
            "small fixed set of justified alternatives for confirmation. Multiple "
            "sets are normal comparison, not permission to optimize against returns.",
            "Use the host's actual web capability to check primary references for "
            "conventional parameters. Do not infer capability from its brand. The "
            "server does not search the web. Record only sources actually retrieved "
            "or supplied by the user, including their title, URL, timezone-aware "
            "accessed_at, retrieval_kind and the limited claim each supports.",
            "Without retrieval, do not claim common, verified or optimal settings. "
            "Offer clearly labelled model_proposal alternatives and explain the "
            "uncertainty; if there is no defensible range, ask the user for a range "
            "or whether to use the proposed range. Do not invent references.",
        ],
        "recording": [
            "After checking the selected implementation, save_research_evidence "
            "records a draft. Use its evidence_id in prepare_backtest, "
            "prepare_comparison, prepare_walk_forward or prepare_optimization. "
            "Every correction creates a new record; a prepared plan retains its "
            "own fixed copy. The tool schema is the authoritative field definition.",
            "Write plain-language entry, holding and exit conditions, direction, "
            "warm-up behavior and limitations; include corresponding formulas and "
            "variable units for advanced readers. For ML, state features, fitted "
            "prediction and decision rules honestly, without inventing a simple "
            "indicator formula or implying an illustrative case is representative. "
            "Explain the actual code; the server cannot prove formula/code "
            "equivalence.",
            "Check formulas against the implementation before saving. For stateful "
            "strategies state initial state, rule precedence and mutually exclusive "
            "updates; use guarded conditions or one complete piecewise transition. "
            "Separate rules must not assign contradictory else branches to the same "
            "signal. State window inclusivity, minimum observations and variance "
            "conventions such as ddof when used. Warm-up conditions must describe "
            "missing data, not the opposite. Distinguish initial warm-up from later "
            "missing inputs: describe both the retained internal state and emitted "
            "signal as the code actually handles them. Include indicator definitions "
            "needed to interpret a rule, not only a library function name. "
            "Distinguish raw execution-price inputs "
            "from actual fills after costs. These checks remain the host's "
            "responsibility, not proof supplied by synthetic validation.",
            "Provide presentation descriptions for every effective constructor leaf. "
            "Parameter paths are JSON pointers (escape ~ as ~0 and / as ~1). "
            "The backend resolves values, including defaults and nested arrays; "
            "parameter_provenance labels their origin without resupplying values.",
            "Each declaration covers target_ids × parameter_paths exactly once. "
            "Use base for one execution; candidate_1, candidate_2, etc. in fixed "
            "comparison order. Optimization needs base plus search_<group>_<option> "
            "for each merged search-axis subtree, with one-based indexes in sorted "
            "parameter-key order. These identify axis options, not Cartesian trials. "
            "Missing, duplicate, unknown or overlapping bindings are rejected.",
            "Origins are user (explicit user values), project_default (actual "
            "selected project configuration), reference (cited value claims), or "
            "model_proposal (unverified proposal). Only reference requires "
            "source_claims. Each cited claim must match the bound value including "
            "JSON type. Group differently valued referenced settings separately. "
            "A formula reference alone does not justify a parameter's value.",
            "If preparation returns research_evidence_binding_invalid, inspect "
            "required_parameter_bindings and required_parameter_paths; explain "
            "every returned value, save corrected evidence and prepare again. "
            "Do not change research or strip training requirements to bypass errors.",
        ],
        "confirmation_and_results": [
            "Preparation checks only code/settings/synthetic fixtures. It adds the "
            "backend's versioned execution assumptions to the saved confirmation. "
            "Present the returned plain-language summary and HTML link, then wait "
            "for explicit approval. Do not download history, train or backtest first.",
            "The resulting HTML preserves formulas, origins, sources, limitations "
            "and saved execution assumptions. Report sections are discoverable "
            "through get_report_sections; research is included in standard recipes. "
            "Generate a whole comparison by comparison_job_id even if every "
            "candidate fails. A recorded reference is a client assertion, not proof "
            "of browsing, semantic accuracy, profitability or strategy suitability.",
        ],
        "unsourced_single_execution_example": example.model_dump(mode="json"),
    }
