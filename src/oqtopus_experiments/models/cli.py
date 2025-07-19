from pydantic import Field, field_validator

from .base import BaseConfigModel


class BaseExperimentCLI(BaseConfigModel):
    shots: int = Field(
        default=1024, ge=1, le=100000, description="Number of measurement shots"
    )
    backend: str = Field(
        default="local_simulator", description="Quantum backend to use"
    )
    devices: list[str] = Field(
        default=["qulacs"], description="List of quantum devices"
    )
    parallel: int = Field(
        default=4, ge=1, le=32, description="Number of parallel execution threads"
    )
    experiment_name: str | None = Field(
        default=None, description="Custom experiment name"
    )
    no_save: bool = Field(default=False, description="Skip saving experiment data")
    no_plot: bool = Field(default=False, description="Skip generating plots")
    show_plot: bool = Field(default=False, description="Display plots immediately")
    verbose: bool = Field(default=False, description="Enable verbose output")

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one device must be specified")
        allowed_devices = ["qulacs", "qiskit", "cirq", "braket", "oqtopus"]
        for device in v:
            if device not in allowed_devices:
                raise ValueError(
                    f"Device '{device}' not in allowed devices: {allowed_devices}"
                )
        return v

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed_backends = ["local_simulator", "oqtopus", "qiskit", "cirq", "braket"]
        if v not in allowed_backends:
            raise ValueError(
                f"Backend '{v}' not in allowed backends: {allowed_backends}"
            )
        return v


class CHSHExperimentCLI(BaseExperimentCLI):
    points: int = Field(
        default=20, ge=1, le=100, description="Number of phase points to scan"
    )
    theta_a: float | None = Field(default=None, description="Custom theta_a angle")
    theta_b: float | None = Field(default=None, description="Custom theta_b angle")
    angles: list[float] | None = Field(
        default=None, description="Custom measurement angles"
    )

    @field_validator("angles")
    @classmethod
    def validate_angles(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 4:
            raise ValueError(
                "Angles list must contain exactly 4 values [theta_a0, theta_a1, theta_b0, theta_b1]"
            )
        return v


class RabiExperimentCLI(BaseExperimentCLI):
    amplitude_start: float = Field(
        default=0.0, ge=0.0, description="Starting amplitude value"
    )
    amplitude_end: float = Field(
        default=1.0, le=2.0, description="Ending amplitude value"
    )
    amplitude_points: int = Field(
        default=20, ge=1, le=100, description="Number of amplitude points"
    )

    @field_validator("amplitude_end")
    @classmethod
    def validate_amplitude_range(cls, v: float, info) -> float:
        if "amplitude_start" in info.data:
            start = info.data["amplitude_start"]
            if v <= start:
                raise ValueError("Amplitude end must be greater than amplitude start")
        return v


class T1ExperimentCLI(BaseExperimentCLI):
    delay_start: float = Field(
        default=0.0, ge=0.0, description="Starting delay time (microseconds)"
    )
    delay_end: float = Field(
        default=100.0, description="Ending delay time (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )

    @field_validator("delay_end")
    @classmethod
    def validate_delay_range(cls, v: float, info) -> float:
        if "delay_start" in info.data:
            start = info.data["delay_start"]
            if v <= start:
                raise ValueError("Delay end must be greater than delay start")
        return v


class T2ExperimentCLI(BaseExperimentCLI):
    delay_start: float = Field(
        default=0.0, ge=0.0, description="Starting delay time (microseconds)"
    )
    delay_end: float = Field(
        default=50.0, description="Ending delay time (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )

    @field_validator("delay_end")
    @classmethod
    def validate_delay_range(cls, v: float, info) -> float:
        if "delay_start" in info.data:
            start = info.data["delay_start"]
            if v <= start:
                raise ValueError("Delay end must be greater than delay start")
        return v


class RamseyExperimentCLI(BaseExperimentCLI):
    delay_start: float = Field(
        default=0.0, ge=0.0, description="Starting delay time (microseconds)"
    )
    delay_end: float = Field(
        default=50.0, description="Ending delay time (microseconds)"
    )
    delay_points: int = Field(
        default=20, ge=1, le=100, description="Number of delay points"
    )
    detuning: float = Field(default=0.0, description="Frequency detuning (MHz)")

    @field_validator("delay_end")
    @classmethod
    def validate_delay_range(cls, v: float, info) -> float:
        if "delay_start" in info.data:
            start = info.data["delay_start"]
            if v <= start:
                raise ValueError("Delay end must be greater than delay start")
        return v


class ParityOscillationExperimentCLI(BaseExperimentCLI):
    phase_start: float = Field(
        default=0.0, description="Starting phase value (radians)"
    )
    phase_end: float = Field(
        default=6.283185307179586, description="Ending phase value (radians)"
    )
    phase_points: int = Field(
        default=20, ge=1, le=100, description="Number of phase points"
    )
    num_qubits: int = Field(
        default=3, ge=2, le=10, description="Number of qubits in GHZ state"
    )

    @field_validator("phase_end")
    @classmethod
    def validate_phase_range(cls, v: float, info) -> float:
        if "phase_start" in info.data:
            start = info.data["phase_start"]
            if v <= start:
                raise ValueError("Phase end must be greater than phase start")
        return v
