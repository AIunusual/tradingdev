"""Pin three historical KD figures and the loader's rejection contract.

The three figures come from the versioned fixture README of the source
branch; they were derived from the named source pickles, not from any
function in this repository. They are historical observations: reproducing
them proves the statistics modules can re-read that data, not that the
current engine would produce the same run.

Tolerance: rel=1e-9. The figures were computed in float64 from the same
committed bytes; 1e-9 is far above float64 accumulation noise for these
magnitudes (notional ~5e6, so the absolute allowance is ~0.005) and far
below any economically meaningful change. A fixture edit also has to pass
the SHA-256 manifest, so a silent drift cannot pass either way.

The fixture's `run.json` also carries the engine's own reported
`engine_metrics`; those are historical provenance only, not the truth for a
recomputed statistic. The named cost-share figure is a specific estimate
(`notional × (fees + slippage) / abs(last equity − first equity)`), not
realised fees or P&L attribution. Daily returns go through
`period_returns(..., "daily")`, so the UTC resample is actually exercised.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from tests.domain.statistics.kd_fixture import (
    FIXTURE_DIR,
    KDFixture,
    load_kd_fixture,
)

from tradingdev.domain.statistics.breakeven import summarize_breakeven
from tradingdev.domain.statistics.performance import summarize_performance
from tradingdev.domain.statistics.periods import period_returns
from tradingdev.domain.statistics.sharpe import sharpe_ratio

if TYPE_CHECKING:
    from pathlib import Path

_REL = 1e-9
_EXPECTED_SHARPE = -7.565103143673017
_EXPECTED_NOTIONAL = 5_014_735.715506879
_EXPECTED_COST_SHARE = 0.863481526386825


@pytest.fixture(scope="module")
def kd() -> KDFixture:
    return load_kd_fixture()


def _daily_returns(kd: KDFixture) -> pd.Series:
    return period_returns(kd.equity, kd.timestamps, "daily")


def _notional(kd: KDFixture) -> float:
    return summarize_breakeven(
        kd.trades,
        start=kd.timestamps[0],
        end=kd.timestamps[-1],
        capital=kd.init_cash,
    ).traded_notional


def _cost_share(kd: KDFixture) -> float:
    # Named formula: notional * (fees + slippage) / abs(last equity - first).
    # This is a specific estimate, not realised fees or P&L attribution.
    rate = kd.fee_rate + kd.slippage_rate
    return _notional(kd) * rate / abs(float(kd.equity[-1] - kd.equity[0]))


class TestPinnedFigures:
    def test_hourly_bars_resample_to_366_utc_days(self, kd: KDFixture) -> None:
        returns = _daily_returns(kd)

        assert len(kd.equity) == 8761
        assert len(kd.trades) == 412
        assert len(returns) == 366
        assert returns.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
        assert returns.index[-1] == pd.Timestamp("2024-12-31", tz="UTC")

    def test_annualised_daily_sharpe(self, kd: KDFixture) -> None:
        sharpe = sharpe_ratio(_daily_returns(kd).to_numpy(), periods_per_year=365.0)

        assert sharpe == pytest.approx(_EXPECTED_SHARPE, rel=_REL)

    def test_traded_notional(self, kd: KDFixture) -> None:
        # Independent cross-check: the engine's recorded volume already sums
        # size_quote + exit_notional for each closed trade.
        engine_volume = float(
            np.sum([t["size_quote"] + t["exit_notional"] for t in kd.trades])
        )

        assert _notional(kd) == pytest.approx(_EXPECTED_NOTIONAL, rel=_REL)
        assert engine_volume == pytest.approx(_EXPECTED_NOTIONAL, rel=_REL)

    def test_named_cost_share_estimate(self, kd: KDFixture) -> None:
        assert _cost_share(kd) == pytest.approx(_EXPECTED_COST_SHARE, rel=_REL)

    def test_engine_metrics_are_provenance_not_the_source_of_truth(
        self, kd: KDFixture
    ) -> None:
        # Even with every engine-reported metric blanked out, the pinned
        # figures still recompute from raw observations. This proves the
        # tests do not read the engine's own reported numbers.
        blanked = replace(kd, run={**kd.run, "engine_metrics": {}})
        assert sharpe_ratio(_daily_returns(blanked), periods_per_year=365.0) == (
            pytest.approx(_EXPECTED_SHARPE, rel=_REL)
        )
        assert _notional(blanked) == pytest.approx(_EXPECTED_NOTIONAL, rel=_REL)
        assert kd.run["engine_metrics"]  # original is intact and has content


class TestCrossModuleRegression:
    def test_periods_to_sharpe(self, kd: KDFixture) -> None:
        returns = _daily_returns(kd)
        assert sharpe_ratio(returns, periods_per_year=365.0) == pytest.approx(
            _EXPECTED_SHARPE, rel=_REL
        )

    def test_trades_to_breakeven(self, kd: KDFixture) -> None:
        summary = summarize_breakeven(
            kd.trades,
            start=kd.timestamps[0],
            end=kd.timestamps[-1],
            capital=kd.init_cash,
        )
        assert summary.traded_notional == pytest.approx(_EXPECTED_NOTIONAL, rel=_REL)
        assert summary.total_trades == 412

    def test_observations_with_fixed_settings_to_performance(
        self, kd: KDFixture
    ) -> None:
        # Settings are explicit; none is inferred from equity or gaps.
        summary = summarize_performance(
            kd.equity,
            kd.timestamps,
            kd.trades,
            init_cash=kd.init_cash,
            frequency="1h",
            periods_per_year=365.0,
            risk_free_rate=0.0,
        )
        assert summary.n_bars == 8761
        assert summary.n_days == 366
        assert summary.closed_trades.total_trades == 412
        assert summary.marked_trades.total_trades == 412
        # Historical data ends with all trades closed; no phantom open leg.
        assert summary.closed_trades.net_pnl == pytest.approx(
            summary.marked_trades.net_pnl, rel=0, abs=0
        )
        # total_pnl starts from init_cash, unlike observed_equity_change.
        assert summary.total_pnl == pytest.approx(
            float(kd.equity[-1] - kd.init_cash), rel=_REL
        )
        assert summary.observed_equity_change == pytest.approx(
            float(kd.equity[-1] - kd.equity[0]), rel=_REL
        )


class TestLoaderRejects:
    @pytest.fixture
    def broken(self, tmp_path: Path) -> Path:
        target = tmp_path / "fixture"
        shutil.copytree(FIXTURE_DIR, target)
        return target

    def test_checksum_mismatch(self, broken: Path) -> None:
        (broken / "equity.parquet").write_bytes(b"tampered")

        with pytest.raises(ValueError, match="checksum mismatch: equity.parquet"):
            load_kd_fixture(broken)

    def test_missing_file(self, broken: Path) -> None:
        (broken / "run.json").unlink()

        with pytest.raises(ValueError, match="fixture file missing: run.json"):
            load_kd_fixture(broken)

    def test_missing_column(self, broken: Path) -> None:
        bars = pd.read_parquet(broken / "equity.parquet")
        bars.drop(columns=["equity"]).to_parquet(broken / "equity.parquet", index=False)

        with pytest.raises(ValueError, match="schema mismatch"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_wrong_dtype(self, broken: Path) -> None:
        bars = pd.read_parquet(broken / "equity.parquet")
        bars["equity"] = bars["equity"].astype("float32")
        bars.to_parquet(broken / "equity.parquet", index=False)

        with pytest.raises(ValueError, match="schema mismatch"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_timestamp_not_strictly_increasing(self, broken: Path) -> None:
        bars = pd.read_parquet(broken / "equity.parquet")
        bars.loc[5, "timestamp"] = bars.loc[4, "timestamp"]
        bars.to_parquet(broken / "equity.parquet", index=False)

        with pytest.raises(ValueError, match="strictly increasing"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_timestamp_shifted_by_one_hour(self, broken: Path) -> None:
        bars = pd.read_parquet(broken / "equity.parquet")
        bars["timestamp"] = bars["timestamp"] + pd.Timedelta(hours=1)
        bars.to_parquet(broken / "equity.parquet", index=False)

        with pytest.raises(ValueError, match="range is"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_interior_timestamp_off_hourly_grid(self, broken: Path) -> None:
        bars = pd.read_parquet(broken / "equity.parquet")
        # Keep the count, endpoints and strict ordering valid so only the
        # internal hourly alignment can reject the regenerated fixture.
        bars.loc[5, "timestamp"] = pd.Timestamp("2024-01-01 05:30:00", tz="UTC")
        bars.to_parquet(broken / "equity.parquet", index=False)

        with pytest.raises(ValueError, match="complete UTC hourly grid"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_missing_run_key(self, broken: Path) -> None:
        run = json.loads((broken / "run.json").read_text())
        del run["slippage"]
        (broken / "run.json").write_text(json.dumps(run))

        with pytest.raises(ValueError, match="missing keys"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_run_identity_mismatch(self, broken: Path) -> None:
        run = json.loads((broken / "run.json").read_text())
        run["symbol"] = "ETH/USDT"
        (broken / "run.json").write_text(json.dumps(run))

        with pytest.raises(ValueError, match="identity does not match"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))

    def test_open_trade_in_fixture_is_rejected(self, broken: Path) -> None:
        trades = pd.read_parquet(broken / "trades.parquet")
        trades.loc[0, "status"] = "open"
        trades.to_parquet(broken / "trades.parquet", index=False)

        with pytest.raises(ValueError, match="statuses must all be 'closed'"):
            load_kd_fixture(broken, checksums=_checksums_of(broken))


def _checksums_of(directory: Path) -> dict[str, str]:
    """Pin mutated copies so the test reaches the schema/range validation."""
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("equity.parquet", "trades.parquet", "run.json")
    }
