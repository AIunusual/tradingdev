# KD(14,3,3) BTC/USDT 1h 2024 fixture

A small, versioned observation set used to re-read three historical KD
backtest figures offline. It is **historical output**, not proof that the
current engine reproduces the same run. No test here unpickles, calls an
extractor, or touches a private workspace cache.

## Source and provenance

These files are extracted from two historical engine runs named in
`run.json` (`source` and `engine_source`, each with a pickle SHA-256). The extractor checks the two runs agree on the equity curve and timestamps
by value (`np.array_equal`, `DatetimeIndex.equals`) and on the trade count and
four shared trade fields. Trade fields come from the current-engine run, whose 412 closed trades carry
`exit_notional`, `status`, and fees. The original extraction script is kept
as `extract.py` for provenance only and is **not imported or executed** by
any test. It needs both private pickles and is never part of a test run.

`run.json` also keeps 33 `engine_metrics` from the `engine_source` pickle
(`cli_....pkl`, the current-engine run). That run records `max_drawdown` as a
positive magnitude (`0.6437650582961815`). The older `source` pickle
(`cd03bf....pkl`, cited by the original demo) used superseded conventions —
bars counted as days and a negative `max_drawdown` — but those older metrics
are **not** copied into `run.json`. The retained figures are still the
engine's own reported values under its historical definitions, so they are
provenance only and must not be treated as truth for a recomputed statistic.

## Contents and format

Schema version 1 (the constant declared in `kd_fixture.py`), with the
observed structure validated on every load:

- `equity.parquet`: 8,761 UTC hourly bars, `timestamp` as
  `datetime64[ns, UTC]`, `equity` as `float64`, from 2024-01-01 00:00 to
  2024-12-31 00:00. Parquet with zstd keeps float64 and nanosecond
  timestamps bit-exact on round trip; delta and dictionary encoding keep the
  file about 11 KB.
- `trades.parquet`: 412 closed trades with `size_quote`, `entry_price`,
  `exit_price`, `net_pnl`, `fee`, `exit_notional` (float64) and `status`
  (string, all `closed`), about 26 KB.
- `run.json`: identity (`kd_crossover`, `BTC/USDT`, `1h`), parameters,
  `init_cash = 10000.0`, `fees = 0.0006`, `slippage = 0.0005`, `mode`,
  provenance, and the historical `engine_metrics`.

Annualisation (`periods_per_year = 365` for daily Sharpe), bar frequency
(`1h`), and risk-free rate (`0.0`) used by the tests are **test settings**,
not values recovered from the source JSON. The source did not record them as
part of this observation set; they are stated explicitly in the tests.

## How to read

`tests/domain/statistics/kd_fixture.py::load_kd_fixture` verifies the SHA-256
of each file, exact column names and dtypes, strictly increasing UTC
timestamps, the recorded range and identity, finite positive equity, and
that all trades are closed. A failed check raises `ValueError` explaining the
reason and, where applicable, naming the affected file. It reads only
committed Parquet/JSON; there is no code path that unpickles or executes
fixture content.

Three figures are pinned at `rel=1e-9` in
`test_statistics_kd_fixture.py`:

- annualised daily Sharpe, ppy=365: `-7.565103143673017`;
- traded notional (both legs): `5014735.715506879`;
- `notional × (fees + slippage) / abs(last equity − first equity)`:
  `0.863481526386825`.

The last value is a **specifically named estimate**, not realised fees or
P&L attribution. Daily returns come from `period_returns(..., "daily")`, so
the UTC resample is exercised rather than bypassed with stored returns.

`test_statistics_cross_module.py` covers boundaries the historical run does
not: first observation after fees versus declared capital, closed-only
versus marked-open trades, zero trades, missing annualisation, volume-mode
zero baseline, and open trades without a phantom exit leg.

## Safe updates

Do not edit a fixture to make a test pass. Daily tests must not rewrite
these files or any workspace/research output; negative tests copy to
`tmp_path` and clean up automatically.

To regenerate, both private pickles named in `run.json` must be present:

```sh
uv run python tests/domain/statistics/fixtures/kd_btcusdt_1h_2024/extract.py
```

Then update the SHA-256 manifest in `kd_fixture.py` and the documented
range/counts in the same reviewed PR, and justify any figure change from a
newly computed independent value. A checksum change without a documented
source change is a rejection, not a silent refresh.
