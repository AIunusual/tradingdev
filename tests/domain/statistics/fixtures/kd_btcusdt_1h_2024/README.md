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

## Rebuild inputs

`extract.py` reads two private pickles that must be placed in
`workspace/data/processed/cache` at the repository root (not committed):

- `cd03bf19459b48af.pkl` (`source`): supplies the equity curve and timestamps;
  its four shared trade fields (`size_quote`, `entry_price`, `exit_price`,
  `net_pnl`) are checked against the other pickle.
- `cli_5ed5dc0b0579cc80_95a98854a8984f82be488c4bff48902d.pkl`
  (`engine_source`): supplies the written trade table (the shared fields plus
  `fee`, `exit_notional`, `status`) and, from its config snapshot and run
  result, the identity, parameters, settings and reported `engine_metrics` in
  `run.json`.

The script verifies each pickle's SHA-256, that the two equity curves and
timestamps are equal by value, the trade count, and the four shared trade
fields before writing. Both filenames and digests are recorded in `run.json`
and in `extract.py`; a missing or altered pickle is rejected, not guessed.

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
