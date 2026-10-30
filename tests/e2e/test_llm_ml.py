"""Explicit model-driven ML research with independently checked MCP artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from tests.app.test_run_metrics import _publish, _scope
from tests.data_cache import seed_market_cache
from tests.e2e.codex_harness import run_codex
from tests.e2e.llm_client import ToolCall, codex_calls, run_local_model
from tests.e2e.ml_report_evidence import (
    ML_REPORT_REQUEST,
    seed_ml_report_candidates,
    verify_ml_report_outcome,
)
from tests.e2e.workflow_arguments import effective_workflow_calls
from tests.integration.mcp_harness import temporary_mcp_workspace
from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.ml_datasets import MLDatasetStore
from tradingdev.adapters.storage.sqlite import SQLiteStore
from tradingdev.app.run_service import RunService
from tradingdev.domain.ml.leaderboard import signal_fingerprint
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
    "source": "binance_vision",
    "market_type": "spot",
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
PREDICTION_ARGUMENTS = {"offset": 0, "limit": 20}
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
之後 predict，以剛回傳 model_id 與同一 dataset_id 取得預覽，
使用 offset=0、limit=20。檢查 n_rows 是完整資料集筆數，next_offset 可續讀。
再 evaluate_signal，使用同一 model_id、dataset_id，設 window=12、n_quantiles=2、
record=true，讓後端評估全部資料列；不需把預覽陣列傳入或讀完每一頁。
若工具回報錯誤，讀取原因再修正。
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
    """Require linked identities, bounded previews and complete backend evaluation."""
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
    assert predicted.arguments == {**identity, **PREDICTION_ARGUMENTS}, (
        "Prediction identity or preview bounds mismatch"
    )
    prediction = predicted.result["prediction"]
    for key, value in identity.items():
        assert prediction[key] == value, "Prediction result identity mismatch"
        assert evaluated.arguments.get(key) == value, "Evaluation identity mismatch"
    assert prediction["n_rows"] == dataset["rows"]
    assert prediction["horizon"] == 1
    assert prediction["offset"] == 0
    assert prediction["limit"] == prediction["next_offset"] == 20
    for key in ("signal", "forward_returns"):
        assert len(prediction[key]) == 20, "Prediction preview must be bounded"
    assert not {"signal", "forward_returns", "horizon"} & evaluated.arguments.keys(), (
        "Evaluation must use saved identities, not caller arrays or horizon"
    )
    for key, value in {"record": True, "window": 12, "n_quantiles": 2}.items():
        assert evaluated.arguments.get(key) == value
    evaluation = evaluated.result
    for key, value in identity.items():
        assert evaluation[key] == value, "Evaluation result identity mismatch"
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
    assert row["parameters"] == model["parameters"]
    assert row["signal_fingerprint"] == prediction["signal_fingerprint"]
    assert row["signal_fingerprint"] == evaluation["signal_fingerprint"]
    assert row["signal_fingerprint"] == evaluation["recorded"]["signal_fingerprint"]
    return MLEvidence(dataset, model, prediction, evaluation, leaderboard)


def assert_ml_artifacts(workspace: MCPWorkspace, evidence: MLEvidence) -> None:
    """Read actual artifacts and recompute labels and hit rate from seeded prices."""
    paths = WorkspacePaths(workspace.workspace)
    dataset = evidence.dataset
    frame, manifest = MLDatasetStore(paths.ml_datasets).load(dataset["dataset_id"])
    assert manifest["content_hash"] == dataset["content_hash"]
    assert manifest["source"] == "binance_vision"
    assert manifest["market_type"] == "spot"
    parquet = Path(dataset["path"])
    assert parquet.is_relative_to(paths.ml_datasets)
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == dataset["content_hash"]
    seed = ml_market_frame().set_index("timestamp")
    expected = (seed["close"].shift(-1) / seed["close"] - 1).loc[frame["timestamp"]]
    np.testing.assert_allclose(frame["forward_return"], expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(frame["target"], np.sign(expected))
    np.testing.assert_array_equal(
        evidence.prediction["forward_returns"], expected.iloc[:20]
    )

    models = ModelArtifactStore(paths.models)
    artifact = models.load(evidence.model["model_id"])
    assert artifact.n_train_samples == len(frame)
    assert list(artifact.feature_names) == dataset["feature_columns"]
    assert artifact.parameters == evidence.model["parameters"]
    fitted = pickle.loads(models.path_for(evidence.model["model_id"]).read_bytes())
    signal = fitted.predict(frame).to_numpy()
    np.testing.assert_array_equal(signal[:20], evidence.prediction["signal"])
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
    assert entry.parameters == artifact.parameters
    assert entry.signal_fingerprint == signal_fingerprint(signal.tolist())
    assert entry.signal_fingerprint == evidence.prediction["signal_fingerprint"]
    assert not list(paths.runs.glob("*/manifest.json")), "ML research must not backtest"


def seed_ml_markets(workspace: MCPWorkspace) -> None:
    """Provide selected spot prices alongside a different default futures market."""
    seed_market_cache(
        workspace.data_root / "processed", ml_market_frame(), market_type="spot"
    )
    other_market = ml_market_frame()
    other_market["close"] *= 2
    workspace.seed_market(other_market)


@pytest.mark.live_llm
def test_llm_ml_pipeline(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        seed_ml_markets(workspace)
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


@pytest.mark.live_llm
def test_natural_request_discovers_saved_ml_report(pytestconfig: pytest.Config) -> None:
    provider = pytestconfig.getoption("llm_provider")
    assert provider in {"codex", "local"}, "Use scripts/check-llm.sh codex|local"
    with temporary_mcp_workspace() as workspace:
        entries = seed_ml_report_candidates(WorkspacePaths(workspace.workspace))
        turns: list[ModelTurn] = []

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": observe_final,
                "max_tool_calls": 18,
            }
            async with workspace.connect() as client:
                names = list(client.output_schemas)
                assert all(name not in ML_REPORT_REQUEST for name in names), (
                    "Natural user request must not supply tool names"
                )
                print(
                    f"[ml-report] Complete product tool set: {len(names)}",
                    flush=True,
                )
                options["tool_names"] = names
                if provider == "local":
                    return await run_local_model(
                        client,
                        ML_REPORT_REQUEST,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(
                await run_codex(workspace.root, ML_REPORT_REQUEST, **options)
            )

        calls = asyncio.run(run())
        transcript = {
            "calls": [asdict(call) for call in calls],
            "turns": [asdict(turn) for turn in turns],
        }
        (workspace.root / "ml-report-transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            assert len(turns) == 1, "Saved research already provides report inputs"
            verify_ml_report_outcome(workspace, calls, turns[0], entries)
        except (AssertionError, KeyError, ValueError, IndexError) as error:
            pytest.fail(
                f"Incomplete {provider}/ML report discovery: {error}\n"
                + json.dumps(transcript, ensure_ascii=False)
            )
    assert not workspace.root.exists()


@pytest.mark.live_llm
def test_natural_request_discovers_rolling_evidence(
    pytestconfig: pytest.Config,
) -> None:
    """Discover both the scheme menu and saved per-roll evidence without a script."""
    request = (
        "請說明這套系統的滾動重訓有哪些視窗方案，固定長度與持續擴大有何差別。"
        "另外請查已保存的結果 rolling_saved，解釋實際嘗試與成功重訓幾次、"
        "第 2 次嘗試用了多少筆已標記樣本，並說明完整證據保存在哪裡。"
        "只查已有資料，不要重新訓練、回測或修改策略。"
    )
    with temporary_mcp_workspace() as workspace:
        paths = WorkspacePaths(workspace.workspace)
        store = SQLiteStore(paths)
        trace = {
            "scheme": "roll_ex",
            "rtype": "ROLL_EX",
            "step": 5,
            "window_size": None,
            "min_train_bars": 55,
            "predict_lookback": 24,
            "label_horizon": 1,
            "bars": 17,
            "bars_scored": 12,
            "initial_model_available": False,
            "unscored_bars": list(range(5)),
            "rolls": [
                {
                    "index": i,
                    "retrain_bar": 5 * i,
                    "train_start": 0,
                    "train_end": 53 + 5 * i,
                    "test_start": 5 * i,
                    "test_end": min(17, 5 * (i + 1)),
                    "n_train_rows": 52 + 5 * i,
                    "trained": i > 0,
                    "bars_scored": 0 if i == 0 else min(5, 17 - 5 * i),
                }
                for i in range(4)
            ],
        }
        _publish(
            (paths, store, RunService(workspace=paths, store=store)),
            "rolling_saved",
            scopes={
                "fold/0/test": _scope(split="test", metadata={"rolling_trace": trace})
            },
            default_scope="fold/0/test",
        )
        turns: list[ModelTurn] = []
        readonly_tools: set[str] = set()

        async def observe_final(turn: ModelTurn) -> None:
            turns.append(turn)

        async def run() -> list[ToolCall]:
            options: dict[str, Any] = {
                "timeout_seconds": pytestconfig.getoption("llm_timeout"),
                "model": pytestconfig.getoption("llm_model"),
                "user_reply": observe_final,
            }
            async with workspace.connect() as client:
                tools = (await client.session.list_tools()).tools
                names = [tool.name for tool in tools]
                readonly_tools.update(
                    tool.name
                    for tool in tools
                    if tool.annotations is not None
                    and tool.annotations.read_only_hint is True
                )
                assert all(name not in request for name in names)
                options["tool_names"] = names
                if pytestconfig.getoption("llm_provider") == "local":
                    return await run_local_model(
                        client,
                        request,
                        base_url=pytestconfig.getoption("llm_base_url"),
                        temperature=pytestconfig.getoption("llm_temperature"),
                        reasoning_effort=pytestconfig.getoption("llm_reasoning_effort"),
                        **options,
                    )
            return codex_calls(await run_codex(workspace.root, request, **options))

        attempted = asyncio.run(run())
        assert all(call.name in readonly_tools for call in attempted), attempted
        calls = effective_workflow_calls(attempted)
        assert any(
            c.name == "get_ml_contract" and c.result.get("success") for c in calls
        )
        summaries = [
            c.result
            for c in calls
            if c.name == "get_run_metrics" and c.result.get("success")
        ]
        assert (
            summaries and summaries[-1]["metadata"]["rolling_trace"]["roll_count"] == 4
        )
        evidence = [
            c.result
            for c in calls
            if c.name == "get_artifact"
            and c.result.get("success")
            and c.arguments.get("artifact_id") == "rolling_saved:performance_json"
            and "content" in c.result
            and "content_page" in c.result
        ]
        assert evidence and any(
            re.search(r'"n_train_rows"\s*:\s*57\b', page["content"])
            for page in evidence
        )
        assert turns and "57" in turns[-1].text
        print(
            "[rolling-evidence] Final reply for semantic review:\n" + turns[-1].text,
            flush=True,
        )
    assert not workspace.root.exists()
