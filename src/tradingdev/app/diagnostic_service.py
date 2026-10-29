"""Read-only provider composition over integrity-checked saved research data."""

from typing import Literal

from tradingdev.adapters.reporting.maverick_diagnostics import MaverickDiagnostics
from tradingdev.app.contracts.common import ContractModel, ErrorResponse
from tradingdev.app.trade_history_service import HistoryReadError, TradeHistoryService
from tradingdev.domain.diagnostics import (
    DiagnosticContext,
    DiagnosticProvider,
    DiagnosticResult,
)


class RunDiagnostics(ContractModel):
    success: Literal[True] = True
    run_id: str
    scope: str
    diagnostics: list[DiagnosticResult]


class DiagnosticService:
    def __init__(
        self,
        history: TradeHistoryService,
        providers: tuple[DiagnosticProvider, ...] | None = None,
    ) -> None:
        self.history = history
        self.providers = (
            providers if providers is not None else (MaverickDiagnostics(),)
        )

    def get(
        self, run_id: str, scope: str | None = None
    ) -> RunDiagnostics | ErrorResponse:
        try:
            loaded = self.history.load_run(run_id)
            scope_id = scope or loaded.performance.default_scope
            if scope_id not in loaded.performance.scopes:
                return ErrorResponse(
                    success=False,
                    code="unknown_history_scope",
                    error=f"Unknown scope: {scope_id}",
                )
            context = DiagnosticContext(
                loaded.performance.scopes[scope_id],
                loaded.observations.scopes.get(scope_id),
            )
            return RunDiagnostics(
                run_id=run_id,
                scope=scope_id,
                diagnostics=self.diagnose(context),
            )
        except HistoryReadError as exc:
            return ErrorResponse(success=False, code=exc.code, error=str(exc))

    def diagnose(self, context: DiagnosticContext) -> list[DiagnosticResult]:
        return [provider.diagnose(context) for provider in self.providers]
