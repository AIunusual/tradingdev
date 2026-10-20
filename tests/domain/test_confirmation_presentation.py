"""Complete, human-readable projections of fixed settings and actual sample evidence."""

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
from tradingdev.domain.data.requirements import DataRequirement
from tradingdev.domain.data.schemas import DataConfig
from tradingdev.domain.execution import ExecutionManifest, OptimizationSpec
from tradingdev.domain.preflight import PreflightReceipt, PreflightWindow
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
        config["validation"] = validation or {"n_splits": 3}
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
    split = manifest.kind != "backtest"
    checks = ["configuration", "signals", "engine", "serialization"]
    if manifest.strategy_execution.kind == "generated":
        checks.append("signal_contract")
    if manifest.kind == "walk_forward":
        checks.append("fit")
    if manifest.kind == "optimization":
        checks.append("candidate_binding")
    validation = manifest.config_copy().get("validation")
    explicit = (
        validation is not None
        and not validation["purged_split"]
        and all(
            validation[key] is not None
            for key in ("train_start", "train_end", "test_start", "test_end")
        )
    )
    return PreflightReceipt.model_validate(
        {
            "manifest_hash": manifest.manifest_hash,
            "elapsed_seconds": 1.5,
            "sample_bars_requested": 64,
            "sample_bars_used": 64,
            "minimum_history_bars": 20,
            "data_source": "binance_vision",
            "windows": [
                PreflightWindow(
                    role="train" if split else "full",
                    start="2024-01-01T00:00:00Z",
                    end="2024-01-02T07:00:00Z",
                    rows=32 if split else 64,
                ),
                *(
                    [
                        PreflightWindow(
                            role="test",
                            start="2024-07-01T00:00:00Z",
                            end="2024-07-02T07:00:00Z",
                            rows=32,
                        )
                    ]
                    if split
                    else []
                ),
            ],
            "checked_paths": checks,
            "trade_count": 0,
            "trading_path_exercised": False,
            "tested_fold_count": 1 if manifest.kind == "walk_forward" else None,
            "total_fold_count": (1 if explicit else validation["n_splits"])
            if validation is not None
            else None,
            "tested_candidates": 1 if manifest.optimization else None,
            "total_candidates": manifest.optimization.total_combinations
            if manifest.optimization
            else None,
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


@pytest.mark.parametrize("kind", ["backtest", "walk_forward", "optimization"])
def test_chat_summary_is_short_and_html_retains_full_review(kind: Any) -> None:
    document = _document(_manifest(kind))
    summary = render_confirmation_summary(document, mode="signal")
    html = unescape(render_confirmation_html(document))
    assert len(summary.splitlines()) <= 7
    assert len(summary) < 900
    for expected in (
        "BTC/USDT",
        "每根 K 線 1 小時",
        "2024-01-01",
        "2024-12-31",
        "10000",
        "0.06%",
        "0.05%",
        "完整 HTML",
        "於對話回覆",
    ):
        assert expected in summary
    assert "試跑只涵蓋有限樣本" in summary
    assert "閱讀本頁不會啟動回測" in html
    for detail in ("最低歷史資料需求", "試跑沒有觀察到交易", "策略設定 1"):
        assert detail not in summary and detail in html
    if kind == "optimization":
        assert "參數組合總數：3" in summary
        assert "搜尋組 1／候選 3" not in summary
        assert "搜尋組 1／候選 3" in html


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
    assert str(position_size or 200) in summary
    assert "初始資金" not in summary
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
    assert "試跑沒有觀察到交易" in text
    assert "已完成必要檢查，可供使用者確認" in text
    assert "最低歷史資料需求：20" in text
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
    assert "已試跑的參數組合：1 / 4" in text
    assert "試跑未涵蓋全部參數組合" in text
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
    assert "單次投入金額：200（" in text
    assert "此計算方式不使用初始資金" in text
    assert "每月最大虧損門檻：1500.0" in text
    assert "每月最大虧損門檻：1500.0" in render_confirmation_text(
        _document(_manifest())
    )
    assert "此計算方式不使用這項設定" in render_confirmation_text(
        _document(_manifest())
    )


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
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "test_start": "2024-07-01",
            "test_end": "2024-12-31",
            "n_splits": 5,
        },
    )
    text = render_confirmation_text(_document(manifest))
    assert "使用指定日期的一組" in text
    assert "指定日期模式固定為一組，此設定不使用" in text
    assert "訓練開始：2024-01-01" in text
    assert "預定分段組數：1" in text
    assert "已試跑的驗證分段：1 / 1" in text


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
                validation={"n_splits": 2, "train_ratio": 0.6, "expanding": expanding},
            )
        )
    )
    assert "分段方式：依資料筆數自動分段" in text
    assert "自動分段的訓練比例：60%" in text
    assert f"訓練區間逐次擴大：{'是' if expanding else '否'}" in text
    assert "已試跑的驗證分段：1 / 2" in text


@pytest.mark.parametrize("dates", [False, True])
def test_purged_validation_shows_effective_mode_and_ignored_controls(
    dates: bool,
) -> None:
    validation: dict[str, Any] = {
        "n_splits": 2,
        "label_horizon": 5,
        "embargo_bars": 2,
        "purged_split": True,
        "train_ratio": 0.6,
        "expanding": False,
    }
    if dates:
        validation.update(
            train_start="2024-01-01",
            train_end="2024-06-30",
            test_start="2024-07-01",
            test_end="2024-12-31",
        )
    manifest = _manifest("walk_forward", validation=validation)
    text = render_confirmation_text(_document(manifest))
    assert "分段方式：淨化分段：固定起點、逐次擴大訓練區間" in text
    assert "淨化分段不使用指定日期" in text
    assert "淨化分段不使用此比例" in text
    assert "訓練區間逐次擴大：是" in text
    assert "此欄設定值（否）不使用" in text
    assert "預定分段組數：2" in text
    assert "已試跑的驗證分段：1 / 2" in text
    assert "未涵蓋所有驗證分段" in text
    assert "使用指定日期的一組" not in text
    if dates:
        assert "訓練開始：2024-01-01" in text
    with pytest.raises(ValueError, match="fold coverage"):
        build_confirmation_document(
            manifest, _presentation(manifest), _receipt(manifest, total_fold_count=1)
        )


def test_leakage_controls_are_rendered_instead_of_being_swallowed() -> None:
    """Purge settings change what a fold may see, so they need a human view."""
    enabled = render_confirmation_text(
        _document(
            _manifest(
                "walk_forward",
                validation={
                    "label_horizon": 5,
                    "embargo_bars": 2,
                    "purged_split": True,
                },
            )
        )
    )
    assert "標籤前視根數：5 根 K 線" in enabled
    assert "淨化分段：是" in enabled
    assert "額外隔離根數：2 根 K 線" in enabled

    defaults = render_confirmation_text(_document(_manifest("walk_forward")))
    assert "標籤前視根數：未啟用檢查" in defaults
    assert "淨化分段：否" in defaults


@pytest.mark.parametrize("horizon", [0, -1])
def test_disabled_label_audit_is_not_presented_as_safe_labels(horizon: int) -> None:
    text = render_confirmation_text(
        _document(_manifest("walk_forward", validation={"label_horizon": horizon}))
    )
    assert f"未啟用檢查（設定值：{horizon}）" in text
    assert "未啟用不代表標籤安全" in text
    assert "標籤不取自未來資料" not in text
    with pytest.raises(ValueError, match="positive label horizon"):
        _document(
            _manifest(
                "walk_forward",
                validation={"purged_split": True, "label_horizon": horizon},
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        "wrong_manifest",
        "missing_engine",
        "unknown_check",
        "unknown_warning",
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
    elif change == "missing_engine":
        receipt.checked_paths.remove("engine")
    elif change == "unknown_check":
        receipt = receipt.model_copy(
            update={"checked_paths": [*receipt.checked_paths, "invented_check"]}
        )
    elif change == "unknown_warning":
        receipt.warnings.append("unmapped_warning")
    else:
        receipt.total_candidates = 2
    with pytest.raises(ValueError):
        build_confirmation_document(manifest, _presentation(manifest), receipt)


@pytest.mark.parametrize(
    "change", ["data_source", "fold_count", "window_role", "extra_check"]
)
def test_preflight_cannot_claim_other_data_or_unexecuted_paths(change: str) -> None:
    manifest = _manifest("walk_forward")
    receipt = _receipt(manifest)
    if change == "data_source":
        receipt.data_source = "different-provider"
    elif change == "fold_count":
        receipt.total_fold_count = 2
    elif change == "window_role":
        receipt.windows[0].role = "full"
    else:
        receipt.checked_paths.append("candidate_binding")
    with pytest.raises(ValueError):
        build_confirmation_document(manifest, _presentation(manifest), receipt)


def test_only_fold_is_not_presented_as_partial_fold_coverage() -> None:
    manifest = _manifest("walk_forward", validation={"n_splits": 1})
    receipt = _receipt(manifest, warnings=["single_fold_only"])
    document = build_confirmation_document(manifest, _presentation(manifest), receipt)
    text = render_confirmation_text(document)
    assert "已試跑的驗證分段：1 / 1" in text
    assert "未涵蓋所有驗證分段" not in text
    assert "完整期間仍可能遇到" in text


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
