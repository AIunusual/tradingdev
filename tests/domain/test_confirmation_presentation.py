"""Human-readable projections of fixed settings and bounded program evidence."""

from __future__ import annotations

from html import unescape
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import ValidationError

from tradingdev.adapters.presentation.confirmation import (
    render_confirmation_html,
    render_confirmation_summary,
    render_confirmation_text,
)
from tradingdev.app.research import SelectedResearchRevision, resolve_research_config
from tradingdev.domain.data.requirements import DataRequirement
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.execution_assumptions import (
    resolve_execution_assumptions,
    summarize_execution_assumptions,
)
from tradingdev.domain.preflight import PreflightReceipt
from tradingdev.domain.presentation.confirmation import (
    SECTION_TITLES,
    ConfirmationDocument,
    ConfirmationPresentation,
    ConfirmationPresentationError,
    ParameterDescription,
    build_confirmation_document,
    parameter_paths,
    validate_presentation,
)
from tradingdev.domain.research import InheritResearchSpec, NewResearchSpec
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.execution import StrategyExecution
from tradingdev.domain.strategies.loader import StrategyLoader

if TYPE_CHECKING:
    import pandas as pd


def _manifest(
    kind: Literal["backtest", "walk_forward", "optimization"] = "backtest",
    *,
    parameters: dict[str, Any] | None = None,
    bundled: bool = False,
    backtest: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    param_ranges: dict[str, list[Any]] | None = None,
    features: list[dict[str, Any]] | None = None,
) -> ExecutionManifest:
    config: dict[str, Any] = {
        "strategy": {
            "id": "fixture",
            "revision_id": "revision-secret",
            "source_path": "/private/strategy.py",
            "source_hash": "b" * 64,
        },
        "backtest": {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "init_cash": 10000,
            "periods_per_year": 365,
            **(backtest or {}),
        },
        "data": {
            **DataConfig(
                raw_dir="/private/raw", processed_dir="/private/processed"
            ).model_dump(),
            "requirements": DataRequirement.model_validate(
                {
                    "market": {
                        "source": "binance_vision",
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                    },
                    "features": features or [],
                }
            ).model_dump(),
        },
    }
    if kind == "walk_forward":
        config["validation"] = validation or {"mode": "automatic", "n_splits": 3}
    search = (
        OptimizationSpec.model_validate(
            {
                "param_ranges": param_ranges or {"period": [10, 20, 30]},
                "optimization_metric": "sharpe_ratio",
                "train_start": "2024-01-01",
                "train_end": "2024-06-30",
                "test_start": "2024-07-01",
                "test_end": "2024-12-31",
            }
        )
        if kind == "optimization"
        else None
    )
    return ExecutionManifest.create(
        kind=kind,
        config=config,
        strategy_execution=StrategyExecution(
            kind="bundled" if bundled else "generated",
            constructor_kwargs={"period": 20} if parameters is None else parameters,
        ),
        optimization=search,
    )


def _receipt(manifest: ExecutionManifest, **changes: Any) -> PreflightReceipt:
    generated = manifest.strategy_execution.kind == "generated"
    checks = [
        "configuration",
        "execution_capabilities",
        "parameter_binding",
        "manifest_serialization",
    ]
    if generated:
        checks.append("signal_contract")
    if manifest.optimization:
        checks.append("candidate_binding")
    return PreflightReceipt.model_validate(
        {
            "schema_version": 2,
            "manifest_hash": manifest.manifest_hash,
            "elapsed_seconds": 1.5,
            "checked_paths": checks,
            "synthetic_fixture_rows": [80, 240] if generated else [],
            "signal_check": "synthetic_contract_passed"
            if generated
            else "bundled_not_executed",
            "training_requirement": "unknown",
            "parameter_combinations_checked": manifest.optimization.total_combinations
            if manifest.optimization
            else None,
            "not_checked": [
                "historical_data",
                "backtest_engine",
                "training",
                "performance",
            ],
            **changes,
        }
    )


def _presentation(manifest: ExecutionManifest) -> ConfirmationPresentation:
    return ConfirmationPresentation(
        title="均線交叉策略",
        summary="以短期與長期均線交叉決定進出場。",
        parameter_descriptions={
            pointer: ParameterDescription(
                label=f"策略設定 {index}",
                description=f"第 {index} 個策略條件",
                unit="根 K 線",
            )
            for index, pointer in enumerate(parameter_paths(manifest), 1)
        },
    )


def _document(manifest: ExecutionManifest) -> ConfirmationDocument:
    return build_confirmation_document(
        manifest, _presentation(manifest), _receipt(manifest), plan_id="plan-private"
    )


def test_volume_month_baseline_is_visible_in_all_confirmation_formats() -> None:
    document = _document(_manifest(backtest={"mode": "volume", "init_cash": None}))
    for rendered in (
        render_confirmation_summary(document, mode="volume"),
        render_confirmation_text(document),
        unescape(render_confirmation_html(document)),
    ):
        assert "上個有行情月份最後收盤權益" in rendered
        assert "不重計前月損益" in rendered


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_chat_summary_preserves_major_rules_and_html_retains_full_review(
    kind: Any,
) -> None:
    document = _document(_manifest(kind))
    summary = render_confirmation_summary(document, mode="signal")
    html = unescape(render_confirmation_html(document))
    assert len(summary.splitlines()) <= 18
    assert len(summary) < 900
    assert document.title not in summary
    assert document.summary in summary
    for expected in (
        "BTC/USDT",
        "每根 K 線 1 小時",
        "2024-01-01",
        "2024-12-31",
        "10000",
        "0.06%",
        "0.05%",
        "時間邊界：依 K 棒時間戳記篩選，包含起訖兩端（有相符資料時）",
        "完整 HTML",
        "於對話回覆",
        "訊號延後一根 K 棒",
        "下一棒 open",
        "1 表示做多，-1 表示做空",
        "0 表示平倉",
        "當時可用資金及成交成本",
        "相同方向的訊號不加碼",
        "停損 未設定；停利 未設定",
        "一般訊號的滑價反映於成交價",
        "觀測到的 UTC 日報酬",
        "不模擬完整保證金、清算",
    ):
        assert expected in summary
    assert "確認前僅檢查程式" in summary
    assert "stoplimit" not in summary
    assert "部位權重：" not in summary
    assert "每月虧損限制：" not in summary
    assert "停損停利後進場：" not in summary
    assert "閱讀本頁不會啟動回測" in html
    assert "依 K 棒時間戳記篩選，包含起訖兩端（有相符資料時）" in html
    assert "純日期代表當日 00:00:00" in html
    assert "結束日期不自動延伸至當日最後一刻" in html
    for detail in (
        "合成訊號檢查筆數",
        "實際可用日期、筆數與資料缺口尚未驗證",
        "策略設定 1",
    ):
        assert detail not in summary and detail in html
    if kind == "optimization":
        assert "參數組合總數：3" in summary
        assert "搜尋組 1／候選 3" not in summary
        assert "搜尋組 1／候選 3" in html


@pytest.mark.parametrize("inherit", [False, True])
def test_research_origins_and_name_are_bound_to_html_without_chat_naming(
    inherit: bool,
) -> None:
    original = _manifest()
    config = resolve_research_config(
        original.config_copy(),
        NewResearchSpec.model_validate(
            {
                "strategy_outline": "低點買入，回到平均價時賣出。",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "start_date": "2024-01-01",
                "end_date": "2024-07-01",
            }
        ),
    )
    if inherit:
        config = resolve_research_config(
            config,
            InheritResearchSpec(fees=0),
            selected_revision=SelectedResearchRevision(
                "fixture", "revision-secret", config, "c" * 64
            ),
        )
    name = "<script>私人策略名稱</script>"
    config["strategy"]["display_name"] = name
    manifest = ExecutionManifest.create(
        kind="backtest", config=config, strategy_execution=original.strategy_execution
    )
    document = _document(manifest)
    html = render_confirmation_html(document)
    summary = render_confirmation_summary(document, mode="signal")
    fields = {
        field.label: field for section in document.sections for field in section.fields
    }
    assert fields["初始資金"].value_text == "10000.0"
    assert fields["滑價率"].value_text == "0.05%"
    assert "來源" in fields["滑價率"].description
    assert fields["研究構想"].value_text == "低點買入，回到平均價時賣出。"
    assert "不代表後端已證明" in fields["研究構想"].description
    assert "365" in summary and "UTC 每日" in summary
    assert "本金與成本來源" in summary
    assert name not in summary and "私人策略名稱" not in summary
    assert name not in html and "&lt;script&gt;私人策略名稱&lt;/script&gt;" in html
    assert "頁面標題與下方說明由模型提供" in html
    assert "策略名稱與下方說明由模型提供" not in html
    assert "不證明輸入來自真人" in html
    if inherit:
        assert fields["手續費率"].value_text == "0%"
        assert "本次輸入" in fields["手續費率"].description
        assert "沿用所選策略版本" in fields["滑價率"].description
        assert "c" * 64 in html
    else:
        assert fields["手續費率"].value_text == "0.06%"
        assert "專案模擬預設" in fields["手續費率"].description
        assert "專案模擬預設" in fields["滑價率"].description


@pytest.mark.parametrize("position_size", [None, 0, 200, 450])
def test_volume_summary_uses_actual_trade_notional_instead_of_unused_capital(
    position_size: int | None,
) -> None:
    document = _document(
        _manifest(
            backtest={
                "mode": "volume",
                "init_cash": 10000,
                "position_size": position_size,
            }
        )
    )
    summary = render_confirmation_summary(document, mode="volume")
    assert "固定名目金額、逐筆累計損益" in summary
    assert "單次投入金額：" in summary
    assert str(200 if position_size is None else position_size) in summary
    assert "初始資金：" not in summary
    assert "10000" not in summary
    assert "此計算方式不使用初始資金" in render_confirmation_html(document)


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_every_execution_kind_has_complete_equivalent_human_views(kind: Any) -> None:
    manifest = _manifest(kind)
    document = _document(manifest)
    text = render_confirmation_text(document)
    html = unescape(render_confirmation_html(document))
    assert tuple(section.id for section in document.sections) == tuple(SECTION_TITLES)
    for section in document.sections:
        assert section.title in text and section.title in html
        for paragraph in section.paragraphs:
            assert paragraph in text and paragraph in html
        for field in section.fields:
            assert field.label in text and field.label in html
            assert field.value_text in text and field.value_text in html
            assert field.description in text and field.description in html
    assert "尚未執行回測引擎" in text
    assert "已完成必要檢查，可供使用者確認" in text
    assert "合成訊號檢查筆數：80、240" in text
    assert "手續費率：0.06%" in text
    assert "滑價率：0.05%" in text
    assert "每根 K 線 1 小時" in text
    for hidden in (
        manifest.manifest_hash,
        "/private/",
        "plan-private",
        "revision-secret",
        "signal_as_position",
        "periods_per_year",
    ):
        assert hidden not in text
        assert hidden not in html.split("<summary>技術識別與檔案資訊</summary>")[0]
    assert "/private/strategy.py" in html
    assert "策略說明（模型提供）" in text
    assert "script-src 'none'" in html


def test_bundled_nested_fit_and_empty_values_require_exact_pointer_metadata() -> None:
    manifest = _manifest(
        bundled=True,
        parameters={
            "config": {"look/back~period": 20, "thresholds": [0, None], "options": {}},
            "fit_config": {"choices": [], "enabled": False},
        },
    )
    assert parameter_paths(manifest) == [
        "/config/look~1back~0period",
        "/config/options",
        "/config/thresholds/0",
        "/config/thresholds/1",
        "/fit_config/choices",
        "/fit_config/enabled",
    ]
    text = render_confirmation_text(_document(manifest))
    assert "空的設定集合" in text and "空的清單" in text
    assert "look/back" not in text and "fit_config" not in text
    assert "策略設定 3：0 根 K 線" in text
    assert "策略設定 4：未設定 根 K 線" in text


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_label"])
def test_metadata_error_exposes_exact_repair_paths(change: str) -> None:
    manifest = _manifest(parameters={"fast": 10, "slow": 20})
    presentation = _presentation(manifest)
    descriptions = dict(presentation.parameter_descriptions)
    if change == "missing":
        del descriptions["/fast"]
    elif change == "extra":
        descriptions["/unused"] = ParameterDescription(
            label="額外設定", description="沒有此參數"
        )
    else:
        descriptions["/slow"] = descriptions["/fast"]
    presentation = presentation.model_copy(
        update={"parameter_descriptions": descriptions}
    )
    with pytest.raises(ConfirmationPresentationError) as exc:
        validate_presentation(manifest, presentation)
    assert exc.value.required_parameter_paths == ["/fast", "/slow"]


def test_nested_search_candidates_keep_labels_correlated_without_raw_json_keys() -> (
    None
):
    manifest = _manifest(
        "optimization",
        parameters={"thresholds": [10], "settings": {"x": 2, "y": 3}},
        param_ranges={
            "thresholds": [[20], [30, 40]],
            "settings": [{"x": 5, "y": 7}, {"x": 11, "y": 13}],
        },
    )
    assert "/thresholds/1" in parameter_paths(manifest)
    text = render_confirmation_text(_document(manifest))
    assert "參數組合總數：4" in text
    assert "已檢查參數綁定的組合數：4" in text
    assert "不代表已執行候選的歷史回測或比較績效" in text
    assert "搜尋組 1／候選 1：策略設定 1：5" in text
    assert "搜尋組 1／候選 1：策略設定 2：7" in text
    assert "搜尋組 2／候選 2：策略設定 4：40" in text
    assert '"settings"' not in text and '"thresholds"' not in text


@pytest.mark.parametrize("bundled", [False, True])
def test_partial_and_empty_candidates_show_the_actual_merged_constructor_values(
    bundled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NestedStrategy(BaseStrategy):
        def __init__(self, settings: dict[str, int]) -> None:
            self.settings = settings

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.copy()

        def get_parameters(self) -> dict[str, Any]:
            return {"settings": self.settings}

    class BundledNestedStrategy(NestedStrategy):
        def __init__(self, config: dict[str, Any], fit_config: dict[str, Any]) -> None:
            super().__init__(config["settings"])
            self.fit_config = fit_config

    base = {"settings": {"x": 2, "y": 3}}
    manifest = _manifest(
        "optimization",
        bundled=bundled,
        parameters={"config": base, "fit_config": {"training": 9}} if bundled else base,
        param_ranges={"settings": [{"x": 5}, {}]},
    )
    before = manifest.model_dump(mode="json")
    prefix = "/config" if bundled else ""
    assert prefix + "/settings" not in parameter_paths(manifest)
    descriptions = {
        prefix + "/settings/x": ParameterDescription(
            label="短期條件", description="短期設定"
        ),
        prefix + "/settings/y": ParameterDescription(
            label="長期條件", description="長期設定"
        ),
    }
    if bundled:
        descriptions["/fit_config/training"] = ParameterDescription(
            label="訓練設定", description="保持固定"
        )
    presentation = ConfirmationPresentation(
        title="巢狀候選", summary="檢查有效設定", parameter_descriptions=descriptions
    )
    document = build_confirmation_document(manifest, presentation, _receipt(manifest))
    rows = {
        field.label: field.value_text
        for section in document.sections
        if section.id == "optimization"
        for field in section.fields
    }

    # Substitute only strategy discovery/model types; use the real loader's
    # candidate merge, constructor binding and actual constructor invocation.
    loader = StrategyLoader()
    cls = BundledNestedStrategy if bundled else NestedStrategy
    monkeypatch.setattr(loader, "load_class", lambda _cfg: cls)
    monkeypatch.setattr(loader, "_execution_models", lambda _cls, _execution: {})
    cfg = {"id": "kd_crossover" if bundled else "generated_nested"}
    for candidate_index, candidate in enumerate(({"x": 5}, {}), 1):
        strategy = loader.create_from_execution(
            cfg,
            manifest.strategy_execution,
            None,
            parameter_overrides={"settings": candidate},
        )
        actual = strategy.get_parameters()["settings"]
        assert actual == {"x": 5 if candidate_index == 1 else 2, "y": 3}
        assert rows[f"搜尋組 1／候選 {candidate_index}：短期條件"] == str(actual["x"])
        assert rows[f"搜尋組 1／候選 {candidate_index}：長期條件"] == str(actual["y"])
        if bundled:
            assert vars(strategy)["fit_config"] == {"training": 9}
    text = render_confirmation_text(document)
    assert "合併後的完整有效設定" in text
    assert "空的設定集合" not in text
    assert manifest.model_dump(mode="json") == before


def test_optimization_timeout_explains_its_actual_estimation_only_scope() -> None:
    document = _document(_manifest("optimization"))
    text = render_confirmation_text(document)
    html = unescape(render_confirmation_html(document))
    for rendered in (text, html):
        assert "第一組估時計算時限" in rendered
        assert "其他候選與保留區間測試不受此時限限制" in rendered
        assert "單一組合最長執行時間" not in rendered


def test_model_unit_never_changes_the_authoritative_parameter_value() -> None:
    manifest = _manifest(parameters={"fraction": 0.02})
    presentation = ConfirmationPresentation(
        title="策略",
        summary="說明",
        parameter_descriptions={
            "/fraction": ParameterDescription(
                label="比例", description="模型宣告的比例", unit="%"
            )
        },
    )
    text = render_confirmation_text(
        build_confirmation_document(manifest, presentation, _receipt(manifest))
    )
    assert "比例：0.02 %" in text
    assert "比例：2 %" not in text


@pytest.mark.parametrize("value", [None, 0])
def test_volume_engine_default_position_and_unused_settings_are_explicit(
    value: Any,
) -> None:
    manifest = _manifest(
        backtest={"mode": "volume", "init_cash": None, "position_size": value}
    )
    text = render_confirmation_text(_document(manifest))
    if value is None:
        assert "單次投入金額：200（執行政策預設）" in text
    else:
        assert "單次投入金額：0.0" in text
        assert "不開新倉，也不產生零數量交易" in text
        assert "200（" not in text
    assert "此計算方式不使用初始資金" in text
    assert "每月最大虧損門檻：1500.0" in text
    assert "每月最大虧損門檻：1500.0" in render_confirmation_text(
        _document(_manifest())
    )
    assert "此計算方式不使用這項設定" in render_confirmation_text(
        _document(_manifest())
    )


@pytest.mark.parametrize("mode", ["signal", "volume"])
@pytest.mark.parametrize("active_stop", [False, True])
def test_confirmation_renders_the_same_resolver_rules_in_every_view(
    mode: Any, active_stop: bool
) -> None:
    config = {"mode": mode}
    if active_stop:
        config.update(stop_loss=0.05, take_profit=0.1, re_entry_after_sl=False)
    manifest = _manifest(backtest=config)
    assumptions = resolve_execution_assumptions(manifest)
    document = _document(manifest)
    section = next(
        section for section in document.sections if section.id == "execution"
    )
    assert section.paragraphs == tuple(
        f"{item.label}：{item.value}" for item in assumptions.items
    )
    assert len(section.paragraphs) == 14
    assert section.summary_paragraphs == summarize_execution_assumptions(assumptions)
    text = render_confirmation_text(document)
    html = unescape(render_confirmation_html(document))
    summary = render_confirmation_summary(document, mode=mode)
    for paragraph in section.paragraphs:
        assert paragraph in text and paragraph in html
    for paragraph in section.summary_paragraphs:
        assert paragraph in summary
    assert "不證明合成檢查已通過" in text
    assert "每根 K 線調整投入金額" not in text
    assert "依回測引擎預設資金配置" not in text
    # Measured baseline fixtures range from 754 to 976 characters. Keep normal
    # confirmations bounded without deleting the active execution assumptions.
    assert len(summary) < (1050 if active_stop else 900)
    if active_stop:
        assert "停損 5%；停利 10%" in summary
        assert "停損停利後進場：" in summary
        if mode == "signal":
            assert "stoplimit" in summary
            assert "出場滑價為零" in summary
            assert "不使用行情 high/low" in summary
        else:
            assert "跨月不自行解除此等待" in summary
    else:
        assert "停損停利後進場：" not in summary
    if mode == "volume":
        assert "只在新倉時乘上基礎金額" in summary
        assert "持有期間不重平衡" in summary
        assert "帳戶報酬及其年化指標不適用" in summary


def test_confirmation_summary_reads_its_saved_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(_manifest())
    expected = render_confirmation_summary(document, mode="signal")

    def fail_resolution(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Rendering must not resolve current execution policy")

    monkeypatch.setattr(
        "tradingdev.domain.execution_assumptions.resolve_execution_assumptions",
        fail_resolution,
    )
    restored = ConfirmationDocument.model_validate_json(document.model_dump_json())
    assert render_confirmation_summary(restored, mode="signal") == expected


def test_feature_paths_stay_in_technical_details_while_sources_remain_visible() -> None:
    manifest = _manifest(
        features=[
            {
                "type": "custom",
                "source": "local-file",
                "column": "hidden_feature_key",
                "path": "/private/feature.csv",
                "raw_path": "/private/raw_feature.csv",
            }
        ]
    )
    document = _document(manifest)
    text = render_confirmation_text(document)
    html = render_confirmation_html(document)
    assert "額外資料 1類型：自訂資料" in text
    assert "資料提供者：local-file" in text
    assert "/private/feature.csv" not in text and "hidden_feature_key" not in text
    assert "/private/feature.csv" in html and "hidden_feature_key" in html


def test_explicit_validation_does_not_claim_automatic_controls_are_active() -> None:
    manifest = _manifest(
        "walk_forward",
        validation={
            "mode": "explicit",
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "test_start": "2024-07-01",
            "test_end": "2024-12-31",
        },
    )
    text = render_confirmation_text(_document(manifest))
    assert "使用指定日期的一組" in text
    assert "訓練比例" not in text
    assert "訓練區間逐次擴大" not in text
    assert "訓練開始：2024-01-01" in text
    assert "預定分段組數：1" in text
    assert "尚未訓練模型或執行驗證分段" in text


@pytest.mark.parametrize("offset", ["", "+08:00"])
def test_explicit_validation_displays_effective_utc_bounds_in_text_and_html(
    offset: str,
) -> None:
    first_day = 2 if offset else 1
    validation = {
        key: f"2024-01-{day:02d}T03:00:00{offset}"
        for key, day in zip(
            ("train_start", "train_end", "test_start", "test_end"),
            range(first_day, first_day + 4),
            strict=True,
        )
    }
    validation["mode"] = "explicit"
    document = _document(_manifest("walk_forward", validation=validation))
    text = render_confirmation_text(document)
    html = unescape(render_confirmation_html(document))
    expected = {
        "訓練開始": "2024-01-01 00:00:00 UTC（含）",
        "訓練結束": "2024-01-03 00:00:00 UTC（不含）",
        "測試開始": "2024-01-03 00:00:00 UTC（含）",
        "測試結束": "2024-01-05 00:00:00 UTC（不含）",
    }
    section = next(
        section for section in document.sections if section.id == "validation"
    )
    actual = {field.label: field.value_text for field in section.fields}
    assert expected.items() <= actual.items()
    for label, value in expected.items():
        assert f"{label}：{value}" in text
        assert value in html
    for rendered in (text, html):
        assert "日期先轉為 UTC（無時區視為 UTC），再忽略時分秒" in rendered
        assert "包含起訖 UTC 整日" in rendered
        assert f"原始設定：{validation['train_start']}" in rendered


@pytest.mark.parametrize("expanding", [False, True])
def test_plain_automatic_validation_keeps_active_ratio_and_expansion(
    expanding: bool,
) -> None:
    text = render_confirmation_text(
        _document(
            _manifest(
                "walk_forward",
                validation={
                    "mode": "automatic",
                    "n_splits": 2,
                    "train_ratio": 0.6,
                    "expanding": expanding,
                },
            )
        )
    )
    assert "分段方式：依資料筆數自動分段" in text
    assert "自動分段的訓練比例：60%" in text
    assert f"訓練區間逐次擴大：{'是' if expanding else '否'}" in text
    assert "尚未訓練模型或執行驗證分段" in text


def test_purged_validation_renders_only_effective_controls() -> None:
    manifest = _manifest(
        "walk_forward",
        validation={
            "mode": "purged",
            "n_splits": 2,
            "label_horizon": 5,
            "embargo_bars": 2,
        },
    )
    text = render_confirmation_text(_document(manifest))
    assert "分段方式：淨化分段：固定起點、逐次擴大訓練區間" in text
    assert "訓練區間逐次擴大：是" in text
    assert "預定分段組數：2" in text
    assert "標籤前視根數：5 根 K 線" in text
    assert "額外隔離根數：2 根 K 線" in text
    assert "訓練開始" not in text
    assert "訓練比例" not in text
    assert "尚未訓練模型或執行驗證分段" in text
    with pytest.raises(ValueError, match="Extra inputs"):
        build_confirmation_document(
            manifest, _presentation(manifest), _receipt(manifest, total_fold_count=1)
        )


def test_disabled_label_audit_is_not_presented_as_safe_labels() -> None:
    text = render_confirmation_text(_document(_manifest("walk_forward")))
    assert "未啟用檢查（設定值：0）" in text
    assert "未啟用不代表標籤安全" in text
    assert "標籤不取自未來資料" not in text
    assert "額外隔離根數" not in text


@pytest.mark.parametrize(
    "change",
    [
        "wrong_manifest",
        "missing_binding",
        "unknown_check",
        "missing_limitation",
        "candidate_count",
    ],
)
def test_invalid_or_incomplete_preflight_evidence_cannot_make_a_confirmation(
    change: str,
) -> None:
    manifest = _manifest("optimization")
    receipt = _receipt(manifest)
    if change == "wrong_manifest":
        receipt.manifest_hash = "0" * 64
    elif change == "missing_binding":
        receipt.checked_paths.remove("parameter_binding")
    elif change == "unknown_check":
        receipt = receipt.model_copy(
            update={"checked_paths": [*receipt.checked_paths, "invented_check"]}
        )
    elif change == "missing_limitation":
        receipt.not_checked.remove("historical_data")
    else:
        receipt.parameter_combinations_checked = 2
    with pytest.raises(ValueError):
        build_confirmation_document(manifest, _presentation(manifest), receipt)


@pytest.mark.parametrize(
    "change",
    [
        "fabricated_engine",
        "fixture_rows",
        "deferred_generated",
        "extra_candidate_check",
    ],
)
def test_preflight_cannot_claim_unexecuted_or_missing_program_checks(
    change: str,
) -> None:
    manifest = _manifest("walk_forward")
    receipt = _receipt(manifest)
    if change == "fabricated_engine":
        receipt = receipt.model_copy(
            update={"checked_paths": [*receipt.checked_paths, "engine"]}
        )
    elif change == "fixture_rows":
        receipt.synthetic_fixture_rows.pop()
    elif change == "deferred_generated":
        receipt = receipt.model_copy(
            update={
                "signal_check": "deferred_until_training",
                "training_requirement": "required",
                "synthetic_fixture_rows": [],
                "checked_paths": [
                    check
                    for check in receipt.checked_paths
                    if check != "signal_contract"
                ],
            }
        )
    else:
        receipt.checked_paths.append("candidate_binding")
    with pytest.raises(ValueError):
        build_confirmation_document(manifest, _presentation(manifest), receipt)


def test_even_one_fold_is_not_presented_as_having_been_trained_or_backtested() -> None:
    document = _document(
        _manifest("walk_forward", validation={"mode": "automatic", "n_splits": 1})
    )
    text = render_confirmation_text(document)
    assert "預定分段組數：1" in text
    assert "尚未訓練模型或執行驗證分段" in text
    assert "尚未取得或檢查真實行情" in text
    assert "尚未執行回測引擎" in text
    assert "已試跑的驗證分段" not in text


def test_required_training_strategy_discloses_that_signals_are_not_executed() -> None:
    manifest = _manifest("walk_forward", bundled=True)
    receipt = _receipt(
        manifest,
        signal_check="deferred_until_training",
        training_requirement="required",
    )
    text = render_confirmation_text(
        build_confirmation_document(manifest, _presentation(manifest), receipt)
    )
    assert "此策略須先訓練；本次只檢查設定與執行模式" in text
    assert "合成訊號檢查筆數" not in text
    assert "尚未計算績效" in text


def test_unknown_execution_setting_is_not_silently_omitted() -> None:
    manifest = _manifest()
    payload = manifest.model_dump(mode="python", exclude={"manifest_hash"})
    payload["config"]["backtest"]["new_hidden_setting"] = 1
    changed = ExecutionManifest.model_validate(
        {**payload, "manifest_hash": ExecutionManifest._digest(payload)}
    )
    with pytest.raises(ValueError):
        _document(changed)


@pytest.mark.parametrize("change", ["omit", "empty", "retitle"])
def test_saved_document_cannot_omit_or_relabel_required_sections(change: str) -> None:
    payload = _document(_manifest()).model_dump(mode="json")
    if change == "omit":
        payload["sections"].pop()
    elif change == "empty":
        payload["sections"][0]["fields"] = []
    else:
        payload["sections"][0]["title"] = "自訂標題"
    with pytest.raises(ValidationError):
        ConfirmationDocument.model_validate(payload)


def test_html_escapes_model_and_parameter_text_without_scripts() -> None:
    attack = "</p><script>alert(1)</script><img src=x>"
    manifest = _manifest(parameters={"value": attack})
    presentation = ConfirmationPresentation(
        title=attack,
        summary=attack,
        parameter_descriptions={
            "/value": ParameterDescription(
                label=attack, description=attack, unit=attack
            )
        },
    )
    document = build_confirmation_document(manifest, presentation, _receipt(manifest))
    html = render_confirmation_html(document)
    assert attack not in html
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert attack in render_confirmation_text(document)
