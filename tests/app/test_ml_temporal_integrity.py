"""Check actual training rows, label observation intervals and holding periods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.ml_service import MLService
from tradingdev.domain.ml.models.lightgbm_model import LightGBMDirectionModel

if TYPE_CHECKING:
    from pathlib import Path

MODEL_ARGUMENTS: dict[str, Any] = {
    "model_kind": "lightgbm_direction",
    "parameters": {"n_estimators": 8, "max_depth": 3, "n_jobs": 1, "random_state": 42},
}
DATA_ARGUMENTS: dict[str, Any] = {
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "feature_set": "returns",
    "horizon": 3,
}
SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def seed_temporal_workspace(root: Path) -> WorkspacePaths:
    """Seed two assets with different prices and one gap, all inside the test root."""
    paths = WorkspacePaths(root / "workspace")
    paths.ensure()
    position = np.arange(200)
    for number, symbol in enumerate(SYMBOLS):
        close = 100 + 15 * number + 5 * np.sin(position / (5 + number) + number)
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=len(position), freq="h", tz="UTC"
                ),
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 100 + position % 11,
            }
        )
        if number:
            frame = frame.drop(index=range(88, 112))
        path = paths.processed_data / (
            symbol.replace("/", "").lower() + "_1h_2024.parquet"
        )
        frame.to_parquet(path, index=False)
    return paths


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    return seed_temporal_workspace(tmp_path)


def test_label_intervals_keep_original_symbol_dates_before_panel_alignment(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace)
    single = service.build_dataset(symbol="ETH/USDT", **DATA_ARGUMENTS)["dataset"]
    panel = service.build_panel(symbols=SYMBOLS, **DATA_ARGUMENTS)["panel"]
    panel_frame = pd.read_parquet(panel["path"])
    groups = [
        ("ETH/USDT", pd.read_parquet(single["path"])),
        *((str(symbol), group) for symbol, group in panel_frame.groupby("symbol")),
    ]
    observed_gap = False
    for symbol, dataset in groups:
        source = pd.read_parquet(
            workspace.processed_data
            / (symbol.replace("/", "").lower() + "_1h_2024.parquet")
        )
        expected_ends = pd.Series(
            source["timestamp"].shift(-3).to_numpy(), index=source["timestamp"]
        ).loc[dataset["timestamp"]]
        pd.testing.assert_series_equal(
            dataset["label_start"].reset_index(drop=True),
            dataset["timestamp"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            dataset["label_end"].reset_index(drop=True),
            expected_ends.reset_index(drop=True),
            check_names=False,
        )
        observed_gap |= bool(
            (
                (dataset["label_end"] - dataset["label_start"]) > pd.Timedelta(hours=3)
            ).any()
        )
    assert observed_gap, (
        "Fixture must retain a label whose price observation crosses a gap"
    )


@pytest.mark.parametrize("scheme", ["purged_walk_forward", "purged_kfold"])
def test_panel_folds_use_shared_dates_and_actual_nonoverlapping_label_intervals(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    service = MLService(workspace)
    panel = service.build_panel(symbols=SYMBOLS, **DATA_ARGUMENTS)["panel"]
    frame = pd.read_parquet(panel["path"])
    fitted: dict[LightGBMDirectionModel, pd.DataFrame] = {}
    observed: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    original_train = LightGBMDirectionModel.train
    original_predict = LightGBMDirectionModel.predict

    def train(
        model: LightGBMDirectionModel,
        inputs: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        fitted[model] = inputs.copy()
        original_train(model, inputs, eval_df)

    def predict(
        model: LightGBMDirectionModel, inputs: pd.DataFrame
    ) -> pd.Series[float]:
        predictions = original_predict(model, inputs)
        observed.append((fitted[model], inputs.assign(prediction=predictions)))
        return predictions

    monkeypatch.setattr(LightGBMDirectionModel, "train", train)
    monkeypatch.setattr(LightGBMDirectionModel, "predict", predict)
    trained = service.train_model(
        dataset_id=panel["dataset_id"],
        validation=scheme,
        n_folds=3,
        embargo=1,
        min_train_bars=10,
        **MODEL_ARGUMENTS,
    )["model"]
    validation = trained["validation"]
    assert validation["leakage_audit_clean"] is True
    folds = [(train, test) for train, test in observed if len(train) < len(frame)]
    assert len(folds) == len(validation["folds"]) == 3
    assert len(observed) == 4 and len(observed[-1][0]) == len(frame)
    finite_ic_folds = 0
    for (training_inputs, test), reported in zip(
        folds, validation["folds"], strict=True
    ):
        assert set(training_inputs.columns) == set(panel["feature_columns"]) | {
            "target"
        }
        training = frame.loc[training_inputs.index]
        assert reported["train_bars"] == len(training)
        assert reported["test_bars"] == len(test)
        assert set(training["timestamp"]).isdisjoint(test["timestamp"])
        for partition in (training, test):
            assert all(
                set(group["symbol"]) == set(SYMBOLS)
                for _, group in partition.groupby("timestamp")
            )
        if scheme == "purged_walk_forward":
            assert training["timestamp"].max() < test["timestamp"].min()
        # Compare the observed information intervals directly, without using
        # the split builder or trusting the service's leakage audit flag.
        first_test_price = test["label_start"].min()
        last_test_price = test["label_end"].max()
        assert (
            (training["label_end"] < first_test_price)
            | (training["label_start"] > last_test_price)
        ).all()
        correlations = []
        for _, asset in test.groupby("symbol"):
            asset = asset.sort_values("timestamp")
            assert 2 <= len(asset) <= 60
            if asset["prediction"].nunique() > 1:
                correlations.append(asset["prediction"].corr(asset["forward_return"]))
        if correlations:
            finite_ic_folds += 1
            assert reported["ic_mean"] == pytest.approx(float(np.mean(correlations)))
        else:
            assert reported["ic_mean"] is None
    assert finite_ic_folds > 0


def test_panel_training_floor_counts_dates_instead_of_symbol_rows(
    workspace: WorkspacePaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingdev.app import ml_training as module

    service = MLService(workspace)
    panel = service.build_panel(symbols=SYMBOLS, **DATA_ARGUMENTS)["panel"]
    minimum = panel["n_dates"] // 4
    assert 2 * (minimum - DATA_ARGUMENTS["horizon"]) >= minimum

    def refuse_fit(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Unusable distinct-date geometry must fail before model creation")

    monkeypatch.setattr(module, "create_direction_model", refuse_fit)
    with pytest.raises(ValueError, match="training bars.*distinct dates"):
        service.train_model(
            dataset_id=panel["dataset_id"],
            validation="purged_walk_forward",
            n_folds=3,
            min_train_bars=minimum,
            **MODEL_ARGUMENTS,
        )
    assert not list(workspace.models.glob("*.bin"))


@pytest.mark.parametrize("scheme", ["purged_walk_forward", "purged_kfold"])
def test_validation_refuses_forward_filled_label_observations(
    workspace: WorkspacePaths, scheme: str
) -> None:
    service = MLService(workspace)
    panel = service.build_panel(
        symbols=SYMBOLS,
        calendar_mode="union",
        missing_policy="forward_fill",
        **DATA_ARGUMENTS,
    )["panel"]
    assert panel["calendar"]["forward_filled"]["ETH/USDT"] > 0
    with pytest.raises(ValueError, match="intersection.*missing_policy=drop"):
        service.train_model(
            dataset_id=panel["dataset_id"],
            validation=scheme,
            n_folds=3,
            **MODEL_ARGUMENTS,
        )
    assert not list(workspace.models.glob("*.bin"))
    trained = service.train_model(dataset_id=panel["dataset_id"], **MODEL_ARGUMENTS)
    prediction = service.predict(
        dataset_id=panel["dataset_id"], model_id=trained["model"]["model_id"]
    )
    assert prediction["prediction"]["n_rows"] == panel["rows"]


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("tamper", ["returns", "horizon", "row_count"])
def test_dataset_bound_evaluation_checks_inputs_even_without_recording(
    workspace: WorkspacePaths, record: bool, tamper: str
) -> None:
    service = MLService(workspace)
    dataset = service.build_dataset(symbol="BTC/USDT", **DATA_ARGUMENTS)["dataset"]
    realized = pd.read_parquet(dataset["path"])["forward_return"].tolist()
    signal = list(np.arange(len(realized), dtype=float))
    arguments: dict[str, Any] = {
        "dataset_id": dataset["dataset_id"],
        "window": 12,
        "n_quantiles": 2,
        "record": record,
    }
    if tamper == "returns":
        realized = [-value for value in realized]
        diagnostic = "forward_returns do not match"
    elif tamper == "horizon":
        arguments["horizon"] = 1
        diagnostic = "does not match the dataset's horizon"
    else:
        signal, realized = signal[:-1], realized[:-1]
        diagnostic = "would not describe this dataset"
    with pytest.raises(ValueError, match=diagnostic):
        service.evaluate_signal(signal, realized, **arguments)
    assert not (workspace.ml_leaderboard / "leaderboard.jsonl").exists()


def test_evaluation_compounds_only_nonoverlapping_horizon_returns(
    workspace: WorkspacePaths,
) -> None:
    service = MLService(workspace)
    signal = list(np.arange(10, dtype=float))
    realized = [0.1] * 10
    multi = service.evaluate_signal(
        signal, realized, horizon=3, window=2, n_quantiles=2
    )
    single = service.evaluate_signal(
        signal, realized, horizon=1, window=2, n_quantiles=2
    )
    # Two warmup rows, then the 3-bar grid buys only at positions 3, 6 and 9.
    assert multi["simulated"]["long_only_cumulative"] == pytest.approx(1.1**3 - 1)
    assert multi["simulated"]["long_only_mean"] == pytest.approx(0.03)
    assert single["simulated"]["long_only_cumulative"] == pytest.approx(1.1**8 - 1)
