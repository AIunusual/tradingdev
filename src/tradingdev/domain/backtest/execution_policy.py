"""Versioned simulator choices shared by execution and saved explanations."""

from __future__ import annotations

from math import inf, nan

EXECUTION_POLICY_VERSION = "execution-v2"
DEFAULT_VOLUME_POSITION_SIZE = 200.0


def signal_execution_defaults() -> dict[str, object]:
    """Pin the vectorbt 1.2.0 defaults that govern the disclosed semantics.

    This deliberately fixes the documented order, sizing, and stop behavior; it
    does not isolate every vectorbt setting or third-party execution engine.
    Non-finite values are internal vectorbt arguments, never serialized evidence.
    """
    return {
        "size": inf,
        "size_type": "amount",
        "fixed_fees": 0.0,
        "reject_prob": 0.0,
        "min_size": 1e-8,
        "max_size": inf,
        "size_granularity": nan,
        "lock_cash": False,
        "allow_partial": True,
        "raise_reject": False,
        "val_price": inf,
        "accumulate": False,
        "sl_stop": nan,
        "sl_trail": False,
        "tp_stop": nan,
        "stop_entry_price": "close",
        "stop_exit_price": "stoplimit",
        "upon_stop_exit": "close",
        "upon_stop_update": "override",
        "upon_long_conflict": "ignore",
        "upon_short_conflict": "ignore",
        "upon_dir_conflict": "ignore",
        "upon_opposite_entry": "reversereduce",
        "cash_sharing": False,
        "trades_type": "exittrades",
    }
