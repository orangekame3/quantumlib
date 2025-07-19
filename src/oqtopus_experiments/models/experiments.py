import math

from pydantic import Field, field_validator

from .base import BaseExperimentModel


class CHSHParameters(BaseExperimentModel):
    standard_theta_a: float = Field(
        default=0.0, description="Standard theta_a value for CHSH experiment"
    )
    standard_theta_b: float = Field(
        default=0.7853981633974483,
        description="Standard theta_b value for CHSH experiment",
    )
    classical_bound: float = Field(
        default=2.0, description="Classical bound for CHSH inequality"
    )
    theoretical_max: float = Field(
        default=2.8284271247461903,
        description="Theoretical maximum value for CHSH inequality",
    )

    @field_validator("classical_bound")
    @classmethod
    def validate_classical_bound(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Classical bound must be positive")
        return v

    @field_validator("theoretical_max")
    @classmethod
    def validate_theoretical_max(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Theoretical maximum must be positive")
        return v


class RabiParameters(BaseExperimentModel):
    amplitude_range: tuple[float, float] = Field(
        default=(0.0, 1.0), description="Range of amplitudes to sweep"
    )
    amplitude_points: int = Field(
        default=20, ge=1, le=100, description="Number of amplitude points"
    )

    @field_validator("amplitude_range")
    @classmethod
    def validate_amplitude_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        start, end = v
        if start < 0:
            raise ValueError("Amplitude start must be non-negative")
        if end <= start:
            raise ValueError("Amplitude end must be greater than start")
        if end > 2.0:
            raise ValueError("Amplitude end should not exceed 2.0")
        return v


class T1Parameters(BaseExperimentModel):
    delay_range: tuple[float, float] = Field(
        default=(0.0, 100.0), description="Range of delay times to sweep (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )

    @field_validator("delay_range")
    @classmethod
    def validate_delay_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        start, end = v
        if start < 0:
            raise ValueError("Delay start must be non-negative")
        if end <= start:
            raise ValueError("Delay end must be greater than start")
        return v


class T2Parameters(BaseExperimentModel):
    delay_range: tuple[float, float] = Field(
        default=(0.0, 50.0), description="Range of delay times to sweep (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )

    @field_validator("delay_range")
    @classmethod
    def validate_delay_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        start, end = v
        if start < 0:
            raise ValueError("Delay start must be non-negative")
        if end <= start:
            raise ValueError("Delay end must be greater than start")
        return v


class RamseyParameters(BaseExperimentModel):
    delay_range: tuple[float, float] = Field(
        default=(0.0, 50.0), description="Range of delay times to sweep (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )
    detuning: float = Field(default=0.0, description="Frequency detuning (MHz)")

    @field_validator("delay_range")
    @classmethod
    def validate_delay_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        start, end = v
        if start < 0:
            raise ValueError("Delay start must be non-negative")
        if end <= start:
            raise ValueError("Delay end must be greater than start")
        return v


class ParityOscillationParameters(BaseExperimentModel):
    phase_range: tuple[float, float] = Field(
        default=(0.0, 2 * math.pi), description="Range of phases to sweep (radians)"
    )
    phase_points: int = Field(
        default=20, ge=1, le=100, description="Number of phase points"
    )
    num_qubits: int = Field(
        default=3, ge=2, le=10, description="Number of qubits in GHZ state"
    )

    @field_validator("phase_range")
    @classmethod
    def validate_phase_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        start, end = v
        if end <= start:
            raise ValueError("Phase end must be greater than start")
        if end - start > 4 * math.pi:
            raise ValueError("Phase range should not exceed 4π")
        return v

    @field_validator("num_qubits")
    @classmethod
    def validate_num_qubits(cls, v: int) -> int:
        if v < 2:
            raise ValueError("Number of qubits must be at least 2")
        if v > 10:
            raise ValueError(
                "Number of qubits should not exceed 10 for practical purposes"
            )
        return v
