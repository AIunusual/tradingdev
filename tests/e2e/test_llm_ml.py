"""Explicit model-driven ML research with independently checked MCP artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tests.e2e.codex_harness import run_codex
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.domain.ml.registry.artifact_store import ModelArtifactStore
from tradingdev.domain.ml.registry.leaderboard_store import LeaderboardStore

if TYPE_CHECKING:
    from tests.e2e.conversation import ModelTurn
    from tests.integration.mcp_harness import MCPWorkspace

ML_TOOLS = [
    "get_ml_contract",
    "build_dataset",
    "train_model",
    "predict",
    "evaluate_signal",
    "leaderboard",
]
DATASET_ARGUMENTS = {
    "symbol": "BTC/USDT",
    "timeframe": "1h",
    "start_date": "2024-01-01",
    "end_date": "2024-01-10",
    "feature_set": "returns",
    "horizon": 1,
}
MODEL_PARAMETERS = {"n_estimators": 12, "max_depth": 3, "n_jobs": 1, "random_state": 42}
TRAINING_ARGUMENTS = {
    "model_kind": "lightgbm_direction",
    "parameters": MODEL_PARAMETERS,
    "validation": "purged_walk_forward",
    "n_folds": 3,
    "embargo": 1,
    "min_train_bars": 20,
}
IN_SAMPLE_STATEMENT = "本次 predict／evaluate_signal 使用訓練集，屬 in-sample。"
VALIDATION_STATEMENT = (
    "validation 衡量訓練流程的樣本外表現，不是此 artifact 的樣本外預測。"
)
ML_PROMPT = f"""請透過 TradingDev MCP 完成一個小型 ML 研究流程。
先 get_ml_contract，確認以下 feature set、model kind、validation、leaderboard metric
皆由契約提供。再 build_dataset，參數為 {json.dumps(DATASET_ARGUMENTS)}。
行情已預先放入後端快取；不要下載行情或寫入策略，也不必啟動回測確認單。
接著 train_model，dataset_id 使用剛回傳的 ID，其餘參數為
{json.dumps(TRAINING_ARGUMENTS)}。檢查 validation 成功及 leakage_audit_clean=true。
之後 predict，以剛回傳 model_id 與同一 dataset_id 產生訊號。
將 predict 回覆的完整 signal 和 forward_returns 原值傳給 evaluate_signal，
不可取樣、截短、四捨五入、編造或重新計算。設 window=12、n_quantiles=2、horizon=1，
record=true，model_id 與 dataset_id 同上。若工具回報錯誤，讀取原因再修正。
最後 leaderboard(metric="hit_rate", dataset_id=同一資料集ID)，
確認排名中包含這次 recorded.entry_id，
與相同 model_id、dataset_id。讀取 caveats，不要把最高排名當作獲利保證。
完成後回覆 dataset_id、model_id、entry_id，並逐字附上以下兩句以區分證據：
{IN_SAMPLE_STATEMENT}
{VALIDATION_STATEMENT}
只使用 MCP 工具，不使用 shell 或直接讀寫檔案；不要請求已授權研究的額外確認。
"""


def ml_market_frame() -> pd.DataFrame:
    """A small reproducible market with both directions and varying returns."""
    rng = np.random.default_rng(42)
    steps = np.arange(240)
    returns = 0.003 * np.sin(steps / 7) + rng.normal(0, 0.0008, len(steps))
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC"),
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": 100 + rng.uniform(0, 50, len(steps)),
        }
    )


@dataclass(frozen=True)
class MLEvidence:
    dataset: dict[str, Any]
    model: dict[str, Any]
    prediction: dict[str, Any]
    evaluation: dict[str, Any]
    leaderboard: dict[str, Any]


def assert_ml_workflow(calls: list[ToolCall]) -> MLEvidence:
    """Require successful linked responses, exact prediction arrays and ordering."""
    calls = effective_workflow_calls(calls)
    assert all(call.name in ML_TOOLS for call in calls), "Unexpected non-ML tool"
    previous = -1
    ordered: list[ToolCall] = []
    for name in ML_TOOLS:
        index = next(
            (
                index
                for index, call in enumerate(calls)
                if index > previous
                and call.name == name
                and isinstance(call.result, dict)
                and call.result.get("success") is True
            ),
            None,
        )
        assert index is not None, f"Missing successful ordered ML step: {name}"
        ordered.append(calls[index])
        previous = index
    contract, built, trained, predicted, evaluated, ranked = ordered
    for key, expected in (
        ("feature_sets", "returns"),
        ("model_kinds", "lightgbm_direction"),
        ("validation_schemes", "purged_walk_forward"),
    ):
        assert expected in {entry["name"] for entry in contract.result[key]}
    assert "hit_rate" in contract.result["leaderboard_metrics"]
    for key, value in DATASET_ARGUMENTS.items():
        assert built.arguments.get(key) == value, f"Dataset request mismatch: {key}"
        assert built.result["dataset"][key] == value, f"Dataset result mismatch: {key}"
    dataset = built.result["dataset"]
    dataset_id = dataset["dataset_id"]
    assert 100 < dataset["rows"] < 250, "Expected the complete small dataset"
    for key, value in TRAINING_ARGUMENTS.items():
        assert trained.arguments.get(key) == value, f"Training request mismatch: {key}"
    assert trained.arguments.get("dataset_id") == dataset_id
    model = trained.result["model"]
    assert model["dataset_id"] == dataset_id
    assert model["n_train_samples"] == dataset["rows"]
    for key, value in MODEL_PARAMETERS.items():
        assert model["parameters"][key] == value
    validation = model["validation"]
    assert validation["scheme"] == "purged_walk_forward"
    assert validation["artifact_trained_on"] == "all_bars"
    assert validation["leakage_audit_clean"] is True
    assert validation["label_horizon"] == 1 and validation["embargo"] == 1
    assert validation["n_folds"] == len(validation["folds"]) == 3
    assert validation["purged_bars"] >= 3 and validation["embargoed_bars"] >= 3
    assert all(fold["train_bars"] >= 20 for fold in validation["folds"])
    identity = {"model_id": model["model_id"], "dataset_id": dataset_id}
    assert predicted.arguments == identity, "Prediction identity mismatch"
    prediction = predicted.result["prediction"]
    for key, value in identity.items():
        assert prediction[key] == value, "Prediction result identity mismatch"
        assert evaluated.arguments.get(key) == value, "Evaluation identity mismatch"
    assert prediction["n_rows"] == dataset["rows"]
    assert prediction["horizon"] == 1
    for key in ("signal", "forward_returns"):
        assert len(prediction[key]) == dataset["rows"]
        assert evaluated.arguments.get(key) == prediction[key], (
            f"Evaluation must retain every exact predicted {key} value"
        )
    for key, value in {"record": True, "window": 12, "n_quantiles": 2}.items():
        assert evaluated.arguments.get(key) == value
    evaluation = evaluated.result
    assert evaluation["horizon"] == 1 and evaluation["window"] == 12
    assert evaluation["n_observations"] == dataset["rows"]
    assert evaluation["recorded"]["n_candidates"] == 1
    assert ranked.arguments.get("metric") == "hit_rate"
    assert ranked.arguments.get("dataset_id") == dataset_id
    leaderboard = ranked.result
    assert leaderboard["metric"] == "hit_rate"
    assert leaderboard["n_candidates"] == leaderboard["n_ranked"] == 1
    assert leaderboard["caveats"], "Leaderboard must retain interpretation caveats"
    row = leaderboard["ranked"][0]
    assert row["entry_id"] == evaluation["recorded"]["entry_id"]
    assert row["dataset_id"] == dataset_id and row["model_id"] == model["model_id"]
    assert row["n_observations"] == dataset["rows"]
    assert row["value"] == evaluation["metrics"]["hit_rate"]
    return MLEvidence(dataset, model, prediction, evaluation, leaderboard)


def assert_ml_artifacts(workspace: MCPWorkspace, evidence: MLEvidence) -> None:
    """Read actual artifacts and recompute labels and hit rate from seeded prices."""
    paths = WorkspacePaths(workspace.workspace)
    dataset = evidence.dataset
    frame, manifest = MLDatasetStore(paths.ml_datasets).load(dataset["dataset_id"])
    assert manifest["content_hash"] == dataset["content_hash"]
    parquet = Path(dataset["path"])
    assert parquet.is_relative_to(paths.ml_datasets)
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == dataset["content_hash"]
    seed = ml_market_frame().set_index("timestamp")
    expected = (seed["close"].shift(-1) / seed["close"] - 1).loc[frame["timestamp"]]
    np.testing.assert_allclose(frame["forward_return"], expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(frame["target"], np.sign(expected))
    np.testing.assert_array_equal(evidence.prediction["forward_returns"], expected)

    models = ModelArtifactStore(paths.models)
    artifact = models.load(evidence.model["model_id"])
    assert artifact.n_train_samples == len(frame)
    assert list(artifact.feature_names) == dataset["feature_columns"]
    assert artifact.parameters == evidence.model["parameters"]
    fitted = pickle.loads(models.path_for(evidence.model["model_id"]).read_bytes())
    np.testing.assert_array_equal(fitted.predict(frame), evidence.prediction["signal"])
    signal = np.asarray(evidence.prediction["signal"])
    realized = expected.to_numpy()
    scored = (signal != 0) & (realized != 0)
    hit_rate = float((np.sign(signal[scored]) == np.sign(realized[scored])).mean())
    assert evidence.evaluation["metrics"]["hit_rate"] == pytest.approx(hit_rate)
    entries = LeaderboardStore(paths.ml_leaderboard).entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.entry_id == evidence.evaluation["recorded"]["entry_id"]
    assert entry.dataset_id == dataset["dataset_id"]
    assert entry.model_id == evidence.model["model_id"]
    assert entry.n_observations == len(frame)
    assert entry.metric("hit_rate") == pytest.approx(hit_rate)
    assert entry.context.data_fingerprint == dataset["content_hash"]
    assert not list(paths.runs.glob("*/manifest.json")), "ML research must not backtest"


@pytest.mark.live_llm
def test_llm_ml_pipeline(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        workspace.seed_market(ml_market_frame())
        turns: list[ModelTurn] = []

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "tool_names": ML_TOOLS,
                "user_reply": observe_final,
                "max_tool_calls": 16,
            }
            if provider == "codex":
                return codex_calls(
                    await run_codex(workspace.root, ML_PROMPT, **options)
                )
            async with workspace.connect() as client:
                return await run_local_model(
                    client,
                    ML_PROMPT,
                    base_url=pytestconfig.getoption("llm_base_url"),
                    temperature=pytestconfig.getoption("llm_temperature"),
                    reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                    **options,
                )

        calls = asyncio.run(run())
        transcript = {
            "calls": [asdict(call) for call in calls],
            "turns": [asdict(turn) for turn in turns],
        }
        (workspace.root / "ml-transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            evidence = assert_ml_workflow(calls)
            assert_ml_artifacts(workspace, evidence)
            assert len(turns) == 1
            assert IN_SAMPLE_STATEMENT in turns[0].text
            assert VALIDATION_STATEMENT in turns[0].text
            for identifier in (
                evidence.dataset["dataset_id"],
                evidence.model["model_id"],
                evidence.evaluation["recorded"]["entry_id"],
            ):
                assert identifier in turns[0].text
        except AssertionError as error:
            pytest.fail(
                f"Incomplete {provider}/ML workflow: {error}\n"
                + json.dumps(transcript, ensure_ascii=False)
            )
