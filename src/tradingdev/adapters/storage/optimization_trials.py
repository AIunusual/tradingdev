"""Atomic optimization ledger bound to the submitted manifest and loaded dataset."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from sqlite3 import Row
from typing import TYPE_CHECKING

from tradingdev.domain.execution import ManifestError
from tradingdev.domain.optimization.grid_search import (
    finite_metric_value,
    is_better_metric,
    parameter_grid,
)
from tradingdev.domain.optimization.trials import TrialEvidence, TrialRecord
from tradingdev.domain.research_evidence_values import canonical, digest
from tradingdev.domain.research_scope import verify_optimization_observations

if TYPE_CHECKING:
    from sqlite3 import Connection

    from pydantic import JsonValue

    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.domain.execution import ExecutionManifest
    from tradingdev.domain.optimization.trials import TrialState

TEST_POSITION = -1


@dataclass(frozen=True)
class TrialSnapshot:
    """All grid positions in manifest order, plus the separate test entry."""

    trials: tuple[TrialRecord, ...]
    test: TrialRecord
    leading_index: int | None

    @property
    def search_complete(self) -> bool:
        """Every approved combination has complete observations, including nulls."""
        return all(trial.state == "completed" for trial in self.trials)


class OptimizationLedger:
    """Commit outcome and evidence together. Only execution workers write entries."""

    def __init__(
        self,
        store: SQLiteStore,
        job_id: str,
        manifest: ExecutionManifest,
        dataset_id: str | None,
    ) -> None:
        if manifest.optimization is None:
            raise ManifestError("Optimization ledger requires an optimization manifest")
        self._store = store
        self.job_id = job_id
        self.manifest = manifest
        self.search = manifest.optimization
        self.dataset_id = dataset_id
        self.grid = parameter_grid(self.search.param_ranges)

    def initialize(self) -> None:
        """Reserve the full grid atomically before the worker starts any trial."""
        with closing(self._store.connect()) as conn, conn:
            conn.executemany(
                """
                insert into optimization_trials(job_id, position, record, record_hash)
                values (?, ?, ?, ?)
                """,
                (
                    self._row_values(TrialRecord(position=index, state="not_attempted"))
                    for index in range(TEST_POSITION, len(self.grid))
                ),
            )

    def start(self, position: int, *, selected_index: int | None = None) -> None:
        """Record the attempt from the process actually executing this position."""
        if not self.dataset_id:
            raise ManifestError("Optimization trial requires a loaded dataset identity")
        self._transition(
            TrialRecord(
                position=position, state="running", selected_index=selected_index
            ),
            expected="not_attempted",
        )

    def finish(
        self,
        position: int,
        *,
        selected_index: int | None = None,
        evidence: TrialEvidence | None = None,
        reason: str | None = None,
        timed_out: bool = False,
    ) -> None:
        """Publish observations or a known candidate failure as one transaction."""
        content = None
        state: TrialState = "timed_out" if timed_out else "failed"
        if evidence is not None:
            state = "completed"
            content = evidence.model_dump_json().encode("utf-8")
        target_value = (
            evidence.scope.values.get(self.search.optimization_metric)
            if evidence is not None
            else None
        )
        if target_value is not None and not isinstance(target_value, (int, float)):
            raise ManifestError("Optimization requires a scalar objective")
        record = TrialRecord(
            position=position,
            state=state,
            selected_index=selected_index,
            target_value=target_value,
            reason=reason,
            evidence_hash=sha256(content).hexdigest() if content is not None else None,
        )
        if content is not None:
            # Verify the serialized artifact, including exact schema and binding,
            # before any result becomes visible to another process.
            self._decode_evidence(record, content)
        self._transition(record, expected="running", evidence=content)

    def snapshot(self, *, verify_evidence: bool = False) -> TrialSnapshot | None:
        """Read one consistent ledger generation and derive the finite leader."""
        with closing(self._store.connect()) as conn:
            rows = conn.execute(
                """
                select position, record, record_hash,
                       evidence is not null as has_evidence
                from optimization_trials where job_id = ? order by position
                """,
                (self.job_id,),
            ).fetchall()
        if not rows:
            return None
        if [row["position"] for row in rows] != list(
            range(TEST_POSITION, len(self.grid))
        ):
            raise ManifestError("Optimization ledger does not match the approved grid")
        records = [self._decode_record(row) for row in rows]
        if verify_evidence:
            # Completed entries are immutable. Read each large observation
            # separately and hash it outside the read transaction so polling
            # never holds one SQLite lock across an entire result bundle.
            for record in records:
                if record.state == "completed":
                    self._verify_hash(record, self._read_evidence_content(record))
        test, *trials = records
        leader: int | None = None
        incumbent: float | None = None
        for trial in trials:
            value = finite_metric_value(trial.target_value)
            if is_better_metric(value, incumbent, self.search.direction):
                leader, incumbent = trial.position, value
        snapshot = TrialSnapshot(tuple(trials), test, leader)
        if test.state != "not_attempted" and (
            not snapshot.search_complete
            or leader is None
            or test.selected_index != leader
        ):
            raise ManifestError(
                "Optimization test differs from the complete grid winner"
            )
        return snapshot

    def read_evidence(self, record: TrialRecord) -> TrialEvidence:
        """Read original observations after checking their identity and integrity."""
        return self._decode_evidence(record, self._read_evidence_content(record))

    def _read_evidence_content(self, record: TrialRecord) -> bytes:
        with closing(self._store.connect()) as conn:
            row = self._read_row(conn, record.position)
        content = row["evidence"]
        if self._decode_record(row) != record or not isinstance(content, bytes):
            raise ManifestError("Observation differs from its ledger")
        return content

    def parameters(self, record: TrialRecord) -> dict[str, JsonValue]:
        index = (
            record.selected_index
            if record.position == TEST_POSITION
            else record.position
        )
        if index is None:
            return {}
        if not 0 <= index < len(self.grid):
            raise ManifestError("Optimization position is outside the approved grid")
        return self.grid[index]

    def _transition(
        self,
        record: TrialRecord,
        *,
        expected: TrialState,
        evidence: bytes | None = None,
    ) -> None:
        self.parameters(record)
        with closing(self._store.connect()) as conn, conn:
            conn.execute("begin immediate")
            previous = self._decode_record(self._read_row(conn, record.position))
            if previous.state != expected or (
                expected == "running"
                and record.selected_index != previous.selected_index
            ):
                raise ManifestError("Invalid optimization ledger transition")
            job_id, position, content, content_hash = self._row_values(record)
            conn.execute(
                """
                update optimization_trials set record = ?, record_hash = ?, evidence = ?
                where job_id = ? and position = ?
                """,
                (content, content_hash, evidence, job_id, position),
            )

    def _read_row(self, conn: Connection, position: int) -> Row:
        row = conn.execute(
            """
            select *, evidence is not null as has_evidence from optimization_trials
            where job_id = ? and position = ?
            """,
            (self.job_id, position),
        ).fetchone()
        if not isinstance(row, Row):
            raise ManifestError("Optimization ledger is missing a grid position")
        return row

    def _row_values(self, record: TrialRecord) -> tuple[str, int, str, str]:
        content = record.model_dump_json()
        return self.job_id, record.position, content, self._record_hash(record, content)

    def _record_hash(self, record: TrialRecord, content: str) -> str:
        # The planned grid exists before data access. Only an actual attempt
        # binds a dataset, and pending positions keep their original hash.
        dataset = self.dataset_id if record.state != "not_attempted" else None
        return digest([self.job_id, self.manifest.manifest_hash, dataset, content])

    def _decode_record(self, row: Row) -> TrialRecord:
        content = row["record"]
        record = TrialRecord.model_validate_json(content)
        if self._record_hash(record, content) != row["record_hash"]:
            raise ManifestError("Optimization ledger entry failed its integrity check")
        if record.position != row["position"] or (
            record.evidence_hash is not None
        ) != bool(row["has_evidence"]):
            raise ManifestError(
                "Optimization ledger entry differs from its position or evidence"
            )
        return record

    @staticmethod
    def _verify_hash(record: TrialRecord, content: bytes) -> None:
        if sha256(content).hexdigest() != record.evidence_hash:
            raise ManifestError("Optimization observation failed its integrity check")

    def _decode_evidence(self, record: TrialRecord, content: bytes) -> TrialEvidence:
        self._verify_hash(record, content)
        evidence = TrialEvidence.model_validate_json(content)
        scope = evidence.scope
        is_test = record.position == TEST_POSITION
        metric = self.search.optimization_metric
        if (
            scope.trial_index != (None if is_test else record.position)
            or canonical(scope.parameters) != canonical(self.parameters(record))
            or metric not in scope.values
            or canonical(scope.values[metric]) != canonical(record.target_value)
        ):
            raise ManifestError(
                "Optimization observation differs from its approved trial"
            )
        verify_optimization_observations(
            self.manifest,
            scope,
            evidence.observations,
            split="test" if is_test else "train",
        )
        return evidence
