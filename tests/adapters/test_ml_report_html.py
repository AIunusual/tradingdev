"""Signal evidence stays complete, readable and separate from trading evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import replace
from html import unescape
from typing import Any

import pytest

from tradingdev.adapters.presentation.html import escape, json_text
from tradingdev.adapters.reporting.html import render_report
from tradingdev.adapters.reporting.ml import ML_REPORT_SCRIPT, render_ml_report
from tradingdev.domain.ml.evaluation.catalog import SIGNAL_METRIC_CATALOG
from tradingdev.domain.ml.leaderboard import EvaluationContext, LeaderboardEntry
from tradingdev.domain.ml.report import ML_SAMPLE_PROVENANCE_NOTE, ml_report_block
from tradingdev.domain.presentation.reports import REPORT_TEMPLATE_VERSION

_CONTEXT = EvaluationContext(
    symbol="BTC/USDT",
    timeframe="1d",
    start_date="2024-01-01",
    end_date="2024-12-31",
    horizon=1,
    window=20,
    n_quantiles=5,
    feature_set="momentum",
    code_fingerprint="feature-code-digest",
    data_fingerprint="data-digest",
)


def _entry(entry_id: str = "candidate-a", **metrics: float | None) -> LeaderboardEntry:
    return LeaderboardEntry(
        entry_id=entry_id,
        label=f"候選 {entry_id}",
        context=_CONTEXT,
        metrics=metrics,
        n_observations=240,
        model_id="model-42",
        dataset_id="dataset-24",
        parameters={"max_depth": 3, "window": 99, "nested": {"normalize": True}},
        signal_fingerprint="signal-digest",
    )


def _payload(*entries: LeaderboardEntry) -> dict[str, Any]:
    return {
        "runs": [],
        "sections": ["ml"],
        "commentary": [],
        "ml": {
            **ml_report_block(entries, "ic_mean", allow_parameter_axes=True),
            "source_sha256": "leaderboard-source-digest",
            "entry_ids": [entry.entry_id for entry in entries],
            "template_version": "1",
        },
    }


def _embedded(document: str) -> dict[str, Any]:
    match = re.search(
        r"<script id='report-data' type='application/json'>(.*?)</script>",
        document,
        flags=re.DOTALL,
    )
    assert match is not None
    result: dict[str, Any] = json.loads(match[1])
    return result


def _body(document: str) -> str:
    return document.split("<main>", 1)[1].split("</main>", 1)[0]


def test_ml_only_report_renders_identity_context_definitions_and_evidence() -> None:
    entry = replace(_entry(), metrics=dict.fromkeys(SIGNAL_METRIC_CATALOG, 0.125))
    payload = _payload(entry)
    document = render_report(payload, "ml-report")
    body = _body(document)
    assert "<h1>ML 訊號研究與候選比較</h1>" in body
    assert "<section data-section='ml' id='ml'>" in body
    assert "尚未計入交易成本" in body
    assert "沒有成交紀錄或可交付的權益曲線" in body
    assert "id='scope-controls'" not in body
    assert "id='result-view'" not in body
    for value in (
        entry.label,
        entry.entry_id,
        entry.model_id,
        entry.dataset_id,
        entry.signal_fingerprint,
        entry.context.symbol,
        entry.context.timeframe,
        entry.context.start_date,
        entry.context.end_date,
        entry.context.feature_set,
        entry.context.code_fingerprint,
        entry.context.data_fingerprint,
        "leaderboard-source-digest",
        "max_depth",
        "normalize",
    ):
        assert value in body
    assert "<th>window</th><td>99</td>" in body
    assert "<th>評估視窗（window）</th><td>20</td>" in body
    for definition in SIGNAL_METRIC_CATALOG.values():
        assert escape(definition.description) in body
        assert f"({definition.id})</code>" in body
    assert "下載完整 ML 證據 JSON" in body
    assert "展開完整 ML 證據（原始欄位與完整精度）" in body
    assert "atob(button.dataset.json)" in document
    assert _embedded(document) == payload
    assert "src='http" not in document
    assert 'src="http' not in document


def test_download_preserves_large_integer_parameters_as_original_json_bytes() -> None:
    large_integer = 9007199254740993
    entry = replace(
        _entry(ic_mean=0.0),
        label="精確證據 🧪",
        parameters={
            "large_integer": large_integer,
            "negative_integer": -large_integer,
            "nested": {"values": [large_integer, 0, None, "繁體中文"]},
        },
    )
    payload = _payload(entry)
    document = render_report(payload, "ml-report")
    match = re.search(r"data-ml-download data-json='([^']+)'", document)
    assert match is not None
    downloaded_bytes = base64.b64decode(match[1], validate=True)
    assert downloaded_bytes == json_text(payload["ml"]).encode("utf-8")
    assert str(large_integer).encode() in downloaded_bytes
    assert str(large_integer - 1).encode() not in downloaded_bytes
    assert json.loads(downloaded_bytes)["scopes"][0]["parameters"] == entry.parameters
    assert "JSON.stringify" not in ML_REPORT_SCRIPT
    assert "reportData" not in ML_REPORT_SCRIPT
    assert "Uint8Array.from(atob(button.dataset.json),char=>char.charCodeAt(0))" in (
        ML_REPORT_SCRIPT
    )


@pytest.mark.parametrize("has_runs", [False, True])
def test_sample_provenance_is_unknown_even_with_saved_model_identity(
    has_runs: bool,
) -> None:
    block = _payload(_entry(ic_mean=0.2))["ml"]
    body = render_ml_report(block, has_runs=has_runs)
    introduction = body.split("<h3>研究範圍與參數座標</h3>", 1)[0]
    assert f"<p class='notice'>{escape(ML_SAMPLE_PROVENANCE_NOTE)}</p>" in introduction
    assert "未保存訓練／評估資料的切分證據" in introduction
    assert "無法判定這些評估結果屬於樣本內或樣本外" in introduction
    assert "<details" not in introduction


def test_default_visibility_preserves_directionless_beta_in_expandable_group() -> None:
    document = render_report(_payload(_entry(ic_mean=0.2, beta=0.5)), "ml-report")
    body = _body(document)
    visible = body.split("<h3>預設比較指標</h3>", 1)[1].split(
        "<details data-ml-hidden-metrics>", 1
    )[0]
    hidden = body.split("<details data-ml-hidden-metrics>", 1)[1].split(
        "</details>", 1
    )[0]
    assert "(ic_mean)" in visible
    assert "(beta)" not in visible
    assert "(beta)" in hidden
    assert "無選優方向（僅描述，不排名）" in hidden
    assert "Beta 表示市場曝險，不代表越大越好" in hidden
    assert "data-ml-hidden-metrics open" not in body


def test_zero_null_and_absent_candidates_keep_distinct_states_and_coordinates() -> None:
    entries = (
        replace(_entry("zero", ic_mean=0.0), context=replace(_CONTEXT, window=10)),
        _entry("null", ic_mean=None),
        replace(_entry("absent", beta=0.2), context=replace(_CONTEXT, window=30)),
    )
    document = render_report(_payload(*entries), "ml-report")
    body = _body(document)
    assert "data-ml-state='value'>0.0</span>" in body
    assert "data-ml-state='null'>不可用（null；含非有限值）</span>" in body
    assert "data-ml-state='absent'>未記錄（absent）</span>" in body
    ranking = body.split("<h3>候選排名與未排名原因</h3>", 1)[1].split(
        "<h3>選優限制與保存警語</h3>", 1
    )[0]
    assert "3 個候選，1 個有有效排序值" in ranking
    assert "<td>1</td><th>" in ranking
    assert ranking.count("<td>未排名</td>") == 2
    assert "<td>20</td>" in ranking
    assert "<td>30</td>" in ranking
    assert escape("'ic_mean' is unavailable or not finite") in ranking
    assert escape("no 'ic_mean' recorded") in ranking
    assert _embedded(document)["ml"]["scopes"][2]["values"] == {"beta": 0.2}


@pytest.mark.parametrize("entries", [(), (_entry(ic_mean=None),)])
def test_no_rankable_candidate_keeps_available_evidence(
    entries: tuple[LeaderboardEntry, ...],
) -> None:
    payload = _payload(*entries)
    document = render_report(payload, "ml-report")
    assert "沒有可排名的候選" in _body(document)
    assert "下載完整 ML 證據 JSON" in _body(document)
    assert _embedded(document) == payload


def test_cross_horizon_caveats_and_selection_limits_are_visible() -> None:
    first = _entry(ic_mean=0.2)
    second = replace(
        _entry("candidate-b", ic_mean=0.3),
        context=replace(_CONTEXT, horizon=2, data_fingerprint="data-horizon-2"),
        dataset_id="dataset-horizon-2",
    )
    payload = _payload(first, second)
    body = _body(render_report(payload, "ml-report"))
    assert "跨 horizon 比較：來源與原始行情等價性未經驗證" in body
    assert "排名本身不證明樣本外有效性" in body
    assert "排序位置依保存的排序順序；同分先後不代表顯著差異" in body
    for caveat in payload["ml"]["caveats"]:
        assert f"<li>{escape(caveat)}</li>" in body


def test_mixed_report_keeps_ml_in_overview_and_run_details_separate() -> None:
    payload = _payload(_entry(ic_mean=0.2))
    payload["sections"].append("metrics")
    payload["runs"] = [
        {
            "identity": {"run_id": "independent-run", "strategy_id": "rule-strategy"},
            "scopes": [
                {
                    "scope_id": "full",
                    "kind": "backtest",
                    "metadata": {},
                    "values": {},
                    "definitions": {},
                    "diagnostics": [],
                }
            ],
        }
    ]
    body = _body(render_report(payload, "mixed-report"))
    overview, details = body.split("<div id='result-view' hidden>", 1)
    assert "<section data-section='ml' id='ml'>" in overview
    assert "<section data-section='ml'" not in details
    assert "<section data-section='metrics'>" in details
    assert "independent-run" in details
    assert "並列不代表這些模型產生了任何回測執行的訊號或交易" in overview


def test_untrusted_ml_fields_are_escaped_in_html_and_embedded_json() -> None:
    attack = "</script><img src=x onerror=alert('unsafe')>"
    entry = replace(
        _entry(attack, ic_mean=0.0),
        label=attack,
        model_id=attack,
        dataset_id=attack,
        signal_fingerprint=attack,
        context=replace(_CONTEXT, symbol=attack, feature_set=attack),
        parameters={attack: {"value": attack}},
    )
    payload = _payload(entry)
    payload["ml"]["caveats"].append(attack)
    payload["ml"]["source_sha256"] = attack
    payload["ml"]["definitions"]["ic_mean"]["description"] = attack
    document = render_report(payload, "ml-report")
    assert attack not in document
    assert escape(attack) in _body(document)
    assert "<img" not in document
    assert "id='ml-candidate-0'" in document
    assert _embedded(document) == payload
    assert attack in unescape(_body(document))


def test_report_without_ml_keeps_template_24_html_bytes_unchanged() -> None:
    # Captured from template 24 after adding the benchmark section. This also
    # protects the unselected-sections list when the catalogue grows.
    payload = {"runs": [], "sections": ["limitations"], "commentary": []}
    document = render_report(payload, "legacy-report")
    assert REPORT_TEMPLATE_VERSION == "24"
    assert hashlib.sha256(document.encode()).hexdigest() == (
        "97872c12410d1d4c7a2ee8a2227e9c3624a2097fd43295dddbe3719e0489e889"
    )
    assert "data-ml-download" not in document
    assert "#ml .ml-candidate" not in document


def test_report_without_ml_pins_interpretation_template_bytes() -> None:
    # Template 24 combines the benchmark section with interpretation/diagnostics
    # rendering; ML remains optional and must stay absent from this request.
    payload = {"runs": [], "sections": ["limitations"], "commentary": []}
    document = render_report(payload, "legacy-report")
    assert REPORT_TEMPLATE_VERSION == "24"
    assert hashlib.sha256(document.encode()).hexdigest() == (
        "97872c12410d1d4c7a2ee8a2227e9c3624a2097fd43295dddbe3719e0489e889"
    )
    assert "data-ml-download" not in document
    assert "#ml .ml-candidate" not in document
