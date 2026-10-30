"""Historical job duration uses persisted lifecycle times at every read boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

import pytest

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.app.job_service import JobService
from tradingdev.app.job_store import JobStore

if TYPE_CHECKING:
    from pathlib import Path


class _Clock(datetime):
    instant = datetime(2024, 1, 1, 1, tzinfo=UTC)

    @classmethod
    def now(cls, tz: object = None) -> Self:
        return cls.fromtimestamp(cls.instant.timestamp(), UTC)


@pytest.mark.parametrize(
    "state", ["done", "failed", "cancelled", "estimation_timeout", "optimizing"]
)
def test_job_elapsed_uses_terminal_end_and_active_clock_consistently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path))
    store.create_job(job_id="elapsed")
    store.update_job(
        "elapsed",
        status=state,
        created_at="2024-01-01T00:00:00+00:00",
        ended_at="2024-01-01T00:00:10+00:00" if state != "optimizing" else None,
    )
    service = JobService(job_store=store)
    monkeypatch.setattr(service, "_is_process_alive", lambda _: True)
    monkeypatch.setattr("tradingdev.app.job_service.datetime", _Clock)
    for hour in (1, 2):
        monkeypatch.setattr(_Clock, "instant", datetime(2024, 1, 1, hour, tzinfo=UTC))
        expected = hour * 3600.0 if state == "optimizing" else 10.0
        status, listed = service.get_job_status("elapsed"), service.list_jobs()[0]
        page = service.list_jobs_page().items[0]
        assert (
            status["elapsed_seconds"]
            == listed["elapsed_seconds"]
            == page.elapsed_seconds
            == expected
        )


@pytest.mark.parametrize(
    "ended_at", [None, "bad", "2023-12-31T00:00:00+00:00", "2024-01-01T00:00:10"]
)
def test_legacy_terminal_invalid_time_is_unavailable_not_current_age(
    tmp_path: Path, ended_at: str | None
) -> None:
    store = JobStore(workspace=WorkspacePaths(tmp_path))
    record = store.create_job(job_id="legacy")
    store.store.upsert_job(
        {
            **record,
            "status": "failed",
            "created_at": "2024-01-01T00:00:00+00:00",
            "ended_at": ended_at,
        }
    )
    service = JobService(job_store=store)
    for response in (service.get_job_status("legacy"), service.list_jobs()[0]):
        assert response["elapsed_seconds"] is None
        assert response["elapsed_unavailable_reason"] == "missing_or_invalid_timestamps"
