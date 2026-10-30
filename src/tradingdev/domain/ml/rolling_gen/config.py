"""Strategy configuration for the optional declarative rolling schemes.

The switch lives here and defaults to off.  When it is off, nothing in this
package is imported by the strategies and the classic retraining loop runs
unchanged; the configuration block is the only thing that decides that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingdev.domain.ml.rolling_gen.registry import resolve_rolling_scheme

if TYPE_CHECKING:
    from tradingdev.domain.ml.rolling_gen.scheme import RollingScheme


class RollingSchemeConfig(BaseModel):
    """A scheme name plus any fields the user chooses to override.

    Unknown keys are rejected rather than ignored: a mistyped ``window_sise``
    would otherwise leave the built-in default in place and look like it had
    been applied.
    """

    model_config = ConfigDict(extra="forbid")

    scheme: str = Field(
        default="roll_sd",
        description="Built-in scheme name; used only when rolling_gen is enabled",
    )
    step: int | None = Field(
        default=None,
        ge=1,
        description="Bars between retrains; unset follows the strategy interval",
    )
    window_size: int | None = Field(
        default=None,
        ge=1,
        description="Fixed training length for ROLL_SD; unset follows 10 x lookback",
    )
    min_train_bars: int | None = Field(
        default=None,
        ge=1,
        description="Labelled rows a window needs; unset follows lookback + 10",
    )
    predict_lookback: int | None = Field(
        default=None,
        ge=1,
        description="Feature context per scored bar; unset follows 2 x lookback",
    )

    def resolve(self) -> RollingScheme:
        """Overlay the declared overrides on the built-in scheme of the same name."""
        return resolve_rolling_scheme(self)


class RollingGenConfig(RollingSchemeConfig):
    """The declarative rolling switch and its scheme, as one strategy setting.

    ``enabled`` is the whole safety story: off means the strategy runs its
    classic retraining loop, on means the declared geometry scores the run.
    """

    enabled: bool = Field(
        default=False,
        description="Off by default; off runs the classic rolling retrainer",
    )

    @model_validator(mode="after")
    def validate_enabled_scheme(self) -> Self:
        """Reject an unusable enabled declaration before strategy fitting."""
        if self.enabled:
            self.resolve()
        return self
