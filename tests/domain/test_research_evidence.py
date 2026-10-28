"""Client source claims cannot substitute for exact execution-value bindings."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError
from tests.domain.test_comparison_confirmation import comparison_fixture
from tests.domain.test_confirmation_presentation import _manifest

from tradingdev.domain.research_evidence import (
    BoundResearchEvidence,
    ParameterClaim,
    ParameterProvenanceDeclaration,
    ResearchEvidenceBindingError,
    ResearchEvidenceDraft,
    ResearchEvidenceRecord,
    ResearchReference,
    bind_research_evidence,
    enumerate_parameter_bindings,
)

if TYPE_CHECKING:
    from tradingdev.domain.execution_target import ExecutionTarget


def evidence_draft(target: ExecutionTarget) -> ResearchEvidenceDraft:
    targets = enumerate_parameter_bindings(target)
    paths = sorted(
        {value.parameter_path for binding in targets for value in binding.parameters}
    )
    return ResearchEvidenceDraft.model_validate(
        {
            "presentation": {
                "title": "均線突破研究",
                "summary": "對照固定設定的歷史行為。",
                "parameter_descriptions": {
                    path: {"label": path, "description": "策略參數", "unit": "根"}
                    for path in paths
                },
            },
            "rules": [
                {
                    "rule_id": "entry",
                    "condition": "收盤價高於均線",
                    "action": "進場",
                    "formula": "close > SMA(close, period)",
                    "variables": [
                        {"symbol": "period", "meaning": "回顧根數", "unit": "根"}
                    ],
                }
            ],
            "limits": ["模型提案未經外部來源核對；過去結果不保證未來表現。"],
            "parameter_provenance": [
                {
                    "target_ids": [binding.target_id],
                    "parameter_paths": [
                        value.parameter_path for value in binding.parameters
                    ],
                    "provenance": "model_proposal",
                    "reason": "作為待驗證的研究設定。",
                }
                for binding in targets
                if binding.parameters
            ],
        }
    )


def _reference(*, value: Any = 20, **changes: Any) -> ResearchReference:
    return ResearchReference.model_validate(
        {
            "source_id": "manual",
            "title": "研究手冊",
            "url": "https://example.org/research?chapter=1#formula",
            "accessed_at": "2026-10-28T09:00:00+08:00",
            "retrieval_kind": "host_web",
            "support": "客戶端回報手冊提供公式及週期設定。",
            "claims": [{"claim_id": "period", "parameter": "回顧根數", "value": value}],
            **changes,
        }
    )


def _reference_draft(*, value: Any = 20) -> ResearchEvidenceDraft:
    draft = evidence_draft(_manifest())
    payload = draft.model_dump(mode="json")
    payload["sources"] = [_reference(value=value).model_dump(mode="json")]
    payload["rules"][0]["source_ids"] = ["manual"]
    payload["parameter_provenance"][0].update(
        provenance="reference",
        source_claims=[{"source_id": "manual", "claim_id": "period"}],
    )
    return ResearchEvidenceDraft.model_validate(payload)


def test_saved_record_and_binding_round_trip_with_exact_identity() -> None:
    target = _manifest()
    draft = _reference_draft()
    created_at = datetime(2026, 10, 28, 1, tzinfo=UTC)
    record = ResearchEvidenceRecord.create(
        draft, evidence_id="a" * 32, created_at=created_at
    )
    bound = bind_research_evidence(record, target)
    restored = BoundResearchEvidence.model_validate_json(bound.model_dump_json())
    restored.verify(target)
    assert restored == bound
    assert restored.record.draft == draft
    assert restored.record.created_at == created_at
    assert restored.parameters[0].value == 20
    assert restored.targets[0].manifest_hash == target.manifest_hash
    assert restored.manifest_hash == target.manifest_hash
    assert len({draft, restored.record.draft}) == 1
    assert len({record, restored.record}) == 1
    assert len({bound, restored}) == 1
    record.verify(record.content_hash)
    with pytest.raises(ValueError, match="expected hash"):
        record.verify("0" * 64)


def test_plain_model_proposal_needs_no_fabricated_retrieval() -> None:
    target = _manifest()
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    assert not bound.record.draft.sources
    assert bound.parameters[0].provenance == "model_proposal"
    assert not bound.parameters[0].source_claims


def test_one_declaration_covers_multiple_paths_and_comparison_candidates() -> None:
    target, _receipt = comparison_fixture()
    payload = evidence_draft(target).model_dump(mode="json")
    payload["parameter_provenance"] = [
        {
            "target_ids": [candidate.candidate_id for candidate in target.candidates],
            "parameter_paths": ["/period"],
            "provenance": "user",
            "reason": "使用者選擇的三組研究值；後端不證明輸入者身分。",
        }
    ]
    record = ResearchEvidenceRecord.create(
        ResearchEvidenceDraft.model_validate(payload)
    )
    bound = bind_research_evidence(record, target)
    assert [entry.value for entry in bound.parameters] == [10, 20, 30]
    assert [entry.manifest_hash for entry in bound.targets] == [
        candidate.manifest.manifest_hash for candidate in target.candidates
    ]
    assert bound.manifest_hash != bound.targets[0].manifest_hash
    assert len(record.draft.parameter_provenance) == 1
    bound.verify(target)


def test_multiple_paths_cartesian_declaration_preserves_zero_and_null() -> None:
    target = _manifest(parameters={"first": 0, "second": None, "third": False})
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    assert [entry.value for entry in bound.parameters] == [0, None, False]
    assert len(bound.record.draft.parameter_provenance) == 1


def test_nested_pointers_and_empty_containers_are_not_lost() -> None:
    target = _manifest(parameters={"a~/": {"nested": [0, None, {}, []]}, "empty": {}})
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    assert {entry.parameter_path: entry.value for entry in bound.parameters} == {
        "/a~0~1/nested/0": 0,
        "/a~0~1/nested/1": None,
        "/a~0~1/nested/2": {},
        "/a~0~1/nested/3": [],
        "/empty": {},
    }
    empty = bound.parameters[-1].value
    assert isinstance(empty, dict)
    with pytest.raises(TypeError, match="immutable"):
        empty["mutation"] = 1


@pytest.mark.parametrize("bundled", [False, True])
def test_optimization_covers_each_merged_option_with_retained_defaults(
    bundled: bool,
) -> None:
    parameters: dict[str, Any] = {
        "thresholds": {"high": 80, "low": 20},
        "period": 0,
    }
    if bundled:
        parameters = {"config": parameters, "fit_config": {"seed": 1}}
    target = _manifest(
        "optimization",
        parameters=parameters,
        bundled=bundled,
        param_ranges={"thresholds": [{"high": 90}, {}], "period": [0, 10]},
    )
    record = ResearchEvidenceRecord.create(evidence_draft(target))
    bound = bind_research_evidence(record, target)
    prefix = "/config" if bundled else ""
    values = {
        binding.target_id: {
            entry.parameter_path: entry.value for entry in binding.parameters
        }
        for binding in bound.targets
    }
    assert list(values) == [
        "base",
        "search_1_1",
        "search_1_2",
        "search_2_1",
        "search_2_2",
    ]
    assert values["search_1_1"] == {prefix + "/period": 0}
    assert values["search_1_2"] == {prefix + "/period": 10}
    assert values["search_2_1"] == {
        prefix + "/thresholds/high": 90,
        prefix + "/thresholds/low": 20,
    }
    assert values["search_2_2"] == {
        prefix + "/thresholds/high": 80,
        prefix + "/thresholds/low": 20,
    }
    incomplete = record.draft.model_dump(mode="json")
    incomplete["parameter_provenance"] = incomplete["parameter_provenance"][:1]
    with pytest.raises(ResearchEvidenceBindingError, match="incomplete") as error:
        bind_research_evidence(
            ResearchEvidenceRecord.create(
                ResearchEvidenceDraft.model_validate(incomplete)
            ),
            target,
        )
    assert len(error.value.required_parameter_bindings) == 5
    assert prefix + "/thresholds/low" in error.value.required_parameter_paths


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "overlap",
        "wrong_target",
        "wrong_path",
        "missing_label",
        "extra_label",
    ],
)
def test_exact_coverage_errors_return_complete_repair_requirements(change: str) -> None:
    target = _manifest()
    payload = evidence_draft(target).model_dump(mode="json")
    declarations = payload["parameter_provenance"]
    if change == "missing":
        declarations.clear()
    elif change == "overlap":
        declarations.append(declarations[0].copy())
    elif change == "wrong_target":
        declarations[0]["target_ids"] = ["candidate_1"]
    elif change == "wrong_path":
        declarations[0]["parameter_paths"] = ["/missing"]
    elif change == "missing_label":
        payload["presentation"]["parameter_descriptions"] = {}
    else:
        payload["presentation"]["parameter_descriptions"]["/extra"] = {
            "label": "額外",
            "description": "不存在",
        }
    with pytest.raises(ResearchEvidenceBindingError) as error:
        bind_research_evidence(
            ResearchEvidenceRecord.create(
                ResearchEvidenceDraft.model_validate(payload)
            ),
            target,
        )
    assert error.value.required_parameter_paths == ["/period"]
    assert error.value.required_parameter_bindings == [
        {
            "target_id": "base",
            "manifest_hash": target.manifest_hash,
            "parameters": [{"parameter_path": "/period", "value": 20}],
        }
    ]


@pytest.mark.parametrize("value", [10, True, "20", [20], {"period": 20}])
def test_reference_claim_must_support_the_exact_effective_leaf(value: Any) -> None:
    with pytest.raises(ResearchEvidenceBindingError, match="do not support"):
        bind_research_evidence(
            ResearchEvidenceRecord.create(_reference_draft(value=value)), _manifest()
        )


def test_formula_only_reference_cannot_claim_parameter_support() -> None:
    payload = _reference_draft().model_dump(mode="json")
    payload["sources"][0]["claims"] = []
    with pytest.raises(ValidationError, match="unknown source claim"):
        ResearchEvidenceDraft.model_validate(payload)
    payload["parameter_provenance"][0].update(
        provenance="model_proposal", source_claims=[]
    )
    draft = ResearchEvidenceDraft.model_validate(payload)
    bound = bind_research_evidence(ResearchEvidenceRecord.create(draft), _manifest())
    assert bound.record.draft.rules[0].source_ids == ("manual",)
    assert bound.parameters[0].provenance == "model_proposal"


@pytest.mark.parametrize("kind", ["user", "project_default", "model_proposal"])
def test_nonreference_provenance_cannot_carry_parameter_claims(kind: str) -> None:
    payload = _reference_draft().parameter_provenance[0].model_dump(mode="json")
    payload["provenance"] = kind
    with pytest.raises(ValidationError, match="Only reference"):
        ParameterProvenanceDeclaration.model_validate(payload)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_source",
        "duplicate_claim",
        "unknown_rule_source",
        "unknown_parameter_source",
        "unknown_claim",
        "duplicate_rule",
        "missing_formula",
    ],
)
def test_dangling_ambiguous_or_incomplete_client_metadata_is_rejected(
    change: str,
) -> None:
    payload = _reference_draft().model_dump(mode="json")
    if change == "duplicate_source":
        payload["sources"].append(payload["sources"][0].copy())
    elif change == "duplicate_claim":
        payload["sources"][0]["claims"].append(
            payload["sources"][0]["claims"][0].copy()
        )
    elif change == "unknown_rule_source":
        payload["rules"][0]["source_ids"] = ["missing"]
    elif change == "unknown_parameter_source":
        payload["parameter_provenance"][0]["source_claims"][0]["source_id"] = "missing"
    elif change == "unknown_claim":
        payload["parameter_provenance"][0]["source_claims"][0]["claim_id"] = "missing"
    elif change == "duplicate_rule":
        payload["rules"].append(payload["rules"][0].copy())
    else:
        payload["rules"][0].pop("formula")
    with pytest.raises(ValidationError):
        ResearchEvidenceDraft.model_validate(payload)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://name:secret@example.org",
        "https://name@example.org",
        "https://",
        "https://example.org:invalid",
        "https://example.org/\nattack",
        "https://example.org\\@other.org",
    ],
)
def test_reference_urls_reject_unsafe_schemes_credentials_and_controls(
    url: str,
) -> None:
    with pytest.raises(ValidationError):
        _reference(url=url)


def test_timezone_and_retrieval_kind_are_not_guessed() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _reference(accessed_at="2026-10-28T01:00:00")
    with pytest.raises(ValidationError):
        _reference(retrieval_kind="backend_verified")
    with pytest.raises(ValueError, match="timezone-aware"):
        ResearchEvidenceRecord.create(
            evidence_draft(_manifest()), created_at=datetime(2026, 10, 28)
        )
    assert _reference(retrieval_kind="user_supplied").retrieval_kind == "user_supplied"


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {"x": float("-inf")}, {1: 2}, "x" * 4001]
)
def test_claim_values_require_bounded_finite_json(value: Any) -> None:
    with pytest.raises(ValidationError):
        ParameterClaim(claim_id="claim", parameter="設定", value=value)


def test_nested_claim_containers_and_presentation_are_detached_and_immutable() -> None:
    value = {"nested": [1, 2]}
    source = _reference(value=value)
    value["nested"].append(3)
    assert source.claims[0].value == {"nested": [1, 2]}
    claim_value = source.claims[0].value
    assert isinstance(claim_value, dict)
    nested = claim_value["nested"]
    assert isinstance(nested, list)
    with pytest.raises(TypeError, match="immutable"):
        nested.append(3)
    draft = evidence_draft(_manifest())
    with pytest.raises(TypeError, match="immutable"):
        draft.presentation.parameter_descriptions.clear()
    with pytest.raises(ValidationError, match="frozen"):
        draft.presentation.__setattr__("title", "changed")
    with pytest.raises(ValidationError, match="frozen"):
        draft.rules[0].__setattr__("formula", "changed")


@pytest.mark.parametrize("kind", ["draft", "record", "bound"])
@pytest.mark.parametrize("version", [2, True, "1", 1.0])
def test_unknown_or_noninteger_schema_versions_are_rejected(
    kind: str, version: Any
) -> None:
    target = _manifest()
    record = ResearchEvidenceRecord.create(evidence_draft(target))
    model: Any = (
        record.draft
        if kind == "draft"
        else record
        if kind == "record"
        else bind_research_evidence(record, target)
    )
    payload = model.model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(ValidationError):
        type(model).model_validate(payload)


def test_unsafe_nested_mutation_and_wrong_target_fail_hash_verification() -> None:
    target = _manifest()
    record = ResearchEvidenceRecord.create(evidence_draft(target))
    bound = bind_research_evidence(record, target)
    with pytest.raises(ValueError, match="another execution"):
        bound.verify(_manifest(parameters={"period": 99}))
    changed = record.model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        changed.verify()
    object.__setattr__(bound.record.draft.presentation, "title", "unsafe alteration")
    with pytest.raises(ValueError, match="hash"):
        bound.verify()
    record.verify()  # The bound evidence owns a detached snapshot.
    corrupted = bound.model_dump(mode="json")
    with pytest.raises(ValidationError):
        BoundResearchEvidence.model_validate(corrupted)


def test_formula_markup_is_preserved_as_data_without_execution() -> None:
    payload = evidence_draft(_manifest()).model_dump(mode="json")
    formula = '</p><script>alert("formula")</script> close > SMA(close, period)'
    payload["rules"][0]["formula"] = formula
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(ResearchEvidenceDraft.model_validate(payload)),
        _manifest(),
    )
    assert bound.record.draft.rules[0].formula == formula


def test_empty_constructor_has_no_invented_parameter_requirement() -> None:
    target = _manifest(parameters={})
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    assert not bound.parameters
    assert not bound.targets[0].parameters


def test_text_list_and_json_depth_limits_are_enforced() -> None:
    draft = evidence_draft(_manifest()).model_dump(mode="json")
    draft["limits"] = ["limit"] * 65
    with pytest.raises(ValidationError):
        ResearchEvidenceDraft.model_validate(draft)
    draft["limits"] = ["x" * 4001]
    with pytest.raises(ValidationError):
        ResearchEvidenceDraft.model_validate(draft)
    value: Any = 0
    for _ in range(18):
        value = [value]
    with pytest.raises(ValidationError, match="depth"):
        _reference(value=value)


def test_model_equality_and_hashing_distinguish_json_boolean_and_number() -> None:
    zero = ParameterClaim(claim_id="claim", parameter="設定", value=0)
    false = ParameterClaim(claim_id="claim", parameter="設定", value=False)
    same = ParameterClaim(claim_id="claim", parameter="設定", value=0)
    assert zero != false
    assert zero == same
    assert hash(zero) == hash(same)
    assert len({zero, false, same}) == 2


def test_record_hash_detects_valid_but_changed_source_claim() -> None:
    record = ResearchEvidenceRecord.create(_reference_draft())
    object.__setattr__(record.draft.sources[0].claims[0], "value", 10)
    with pytest.raises(ValueError, match="hash"):
        record.verify()


def test_bound_hash_and_expanded_value_mutation_are_rejected() -> None:
    target = _manifest()
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(evidence_draft(target)), target
    )
    changed = bound.model_copy(update={"content_hash": "0" * 64})
    with pytest.raises(ValueError, match="hash"):
        changed.verify()
    object.__setattr__(bound.parameters[0], "value", 99)
    with pytest.raises(ValueError, match="differs"):
        bound.verify()


def test_parameter_values_never_accept_client_supplied_executable_values() -> None:
    payload = evidence_draft(_manifest()).model_dump(mode="json")
    payload["parameter_provenance"][0]["value"] = 99
    with pytest.raises(ValidationError, match="Extra inputs"):
        ResearchEvidenceDraft.model_validate(payload)


def test_reference_claims_are_bound_to_each_exact_comparison_value() -> None:
    target, _receipt = comparison_fixture()
    payload = evidence_draft(target).model_dump(mode="json")
    source = _reference().model_dump(mode="json")
    source["claims"] = [
        {"claim_id": f"period{period}", "parameter": "回顧根數", "value": period}
        for period in (10, 20, 30)
    ]
    payload["sources"] = [source]
    declaration = {
        "target_ids": ["candidate_1", "candidate_2", "candidate_3"],
        "parameter_paths": ["/period"],
        "provenance": "reference",
        "reason": "客戶端回報手冊的三組值；語意由客戶端負責。",
        "source_claims": [
            {"source_id": "manual", "claim_id": f"period{period}"}
            for period in (10, 20, 30)
        ],
    }
    payload["parameter_provenance"] = [declaration]
    with pytest.raises(ResearchEvidenceBindingError, match="candidate_1"):
        bind_research_evidence(
            ResearchEvidenceRecord.create(
                ResearchEvidenceDraft.model_validate(payload)
            ),
            target,
        )
    payload["parameter_provenance"] = [
        {
            **declaration,
            "target_ids": [f"candidate_{index}"],
            "source_claims": [{"source_id": "manual", "claim_id": f"period{period}"}],
        }
        for index, period in enumerate((10, 20, 30), 1)
    ]
    bound = bind_research_evidence(
        ResearchEvidenceRecord.create(ResearchEvidenceDraft.model_validate(payload)),
        target,
    )
    assert [entry.value for entry in bound.parameters] == [10, 20, 30]
    assert [
        [claim.claim_id for claim in entry.source_claims] for entry in bound.parameters
    ] == [["period10"], ["period20"], ["period30"]]
    payload["parameter_provenance"][2]["source_claims"][0]["claim_id"] = "period20"
    with pytest.raises(ResearchEvidenceBindingError, match="candidate_3"):
        bind_research_evidence(
            ResearchEvidenceRecord.create(
                ResearchEvidenceDraft.model_validate(payload)
            ),
            target,
        )


@pytest.mark.parametrize(
    ("field", "length"),
    [
        ("title", 161),
        ("summary", 4001),
        ("label", 161),
        ("description", 4001),
        ("unit", 161),
        ("path", 1025),
    ],
)
def test_presentation_text_limits_are_visible_in_schema(
    field: str, length: int
) -> None:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError as SchemaValidationError

    payload = evidence_draft(_manifest()).model_dump(mode="json")
    presentation = payload["presentation"]
    if field in {"title", "summary"}:
        presentation[field] = "x" * length
    elif field == "path":
        presentation["parameter_descriptions"] = {
            "/" + "x" * (length - 1): presentation["parameter_descriptions"]["/period"]
        }
    else:
        presentation["parameter_descriptions"]["/period"][field] = "x" * length
    with pytest.raises(ValidationError):
        ResearchEvidenceDraft.model_validate(payload)
    with pytest.raises(SchemaValidationError):
        Draft202012Validator(ResearchEvidenceDraft.model_json_schema()).validate(
            payload
        )
