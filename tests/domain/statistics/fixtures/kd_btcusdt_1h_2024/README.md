# KD(14,3,3) BTC/USDT 1h 2024 fixture

A small, versioned observation set for offline statistics tests. It is
**historical output**, not a replay of the current engine. Tests read only
committed files: no unpickling, extractor execution, or private workspace
cache access.

## Files

- `equity.parquet`: 8,761 UTC hourly bars, `timestamp` (`datetime64[ns, UTC]`)
  and `equity` (`float64`), 2024-01-01 00:00 to 2024-12-31 00:00.
- `trades.parquet`: 412 closed trades with `size_quote`, `entry_price`,
  `exit_price`, `net_pnl`, `fee`, `exit_notional` (all `float64`) and `status`
  (string, all `closed`).
- `run.json`: identity, parameters, `init_cash`, `fees`, `slippage`, `mode`,
  pickle provenance, and the engine's own reported `engine_metrics`.
- `extract.py`: provenance-only rebuild script, never imported by tests.

Schema version is the `SCHEMA_VERSION` constant in `kd_fixture.py`; column
names, dtypes and identities are checked on every load. Which settings the
tests declare rather than recover from `run.json`, which figures are pinned,
and why they are historical observations are documented in
`tests/domain/statistics/test_statistics_kd_fixture.py` and
`tests/domain/statistics/test_statistics_cross_module.py`.

## Update procedure

Do not edit a fixture to make a test pass. Daily tests must not rewrite these
files or any workspace/research output; negative tests copy to `tmp_path` and
clean up automatically.

To regenerate, both private pickles named in `run.json` must be present:

```sh
uv run python tests/domain/statistics/fixtures/kd_btcusdt_1h_2024/extract.py
```

Then update the SHA-256 manifest in `kd_fixture.py` and the documented
file structure in the same reviewed PR. Justify any figure change from a new
independent computation. A checksum change without a documented source change
is a rejection, not a silent refresh.
