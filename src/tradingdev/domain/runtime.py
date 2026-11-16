"""Immutable resource policy and identity for isolated strategy execution."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeLimits(BaseModel):
    """Limits apply to the entire execution unit, including all descendants."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_seconds: float = Field(default=300, gt=0, le=86400, strict=True)
    memory_bytes: int = Field(default=2 * 1024**3, ge=64 * 1024**2, strict=True)
    cpus: float = Field(default=2, gt=0, le=256, strict=True)
    processes: int = Field(default=128, ge=8, le=4096, strict=True)
    temporary_bytes: int = Field(
        default=256 * 1024**2,
        ge=1024**2,
        strict=True,
        description="Temporary storage cannot exceed memory_bytes.",
    )
    message_bytes: int = Field(
        default=64 * 1024**2,
        ge=1024,
        strict=True,
        description="Each message cannot exceed one quarter of memory_bytes.",
    )
    diagnostic_bytes: int = Field(
        default=64 * 1024,
        ge=1024,
        strict=True,
        description="Retained diagnostic bytes cannot exceed message_bytes.",
    )

    @model_validator(mode="after")
    def _bounded_buffers(self) -> Self:
        if self.temporary_bytes > self.memory_bytes:
            raise ValueError("Temporary storage cannot exceed the memory limit")
        if self.message_bytes > self.memory_bytes // 4:
            raise ValueError("A message cannot exceed one quarter of the memory limit")
        if self.diagnostic_bytes > self.message_bytes:
            raise ValueError("Diagnostics cannot exceed the message limit")
        return self


class RuntimeSpec(BaseModel):
    """An immutable image identity and the effective policy selected for a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)


class IsolatedRuntimeError(RuntimeError):
    """A failed isolated operation, including independently verified cleanup."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cleanup_verified: bool = False,
        strategy_callback_failed: bool = False,
        control_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cleanup_verified = cleanup_verified
        self.strategy_callback_failed = strategy_callback_failed
        self.control_path = control_path
