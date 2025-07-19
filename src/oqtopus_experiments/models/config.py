from pydantic import Field, field_validator

from .base import BaseConfigModel
from .experiments import CHSHParameters


class ExperimentSettings(BaseConfigModel):
    default_shots: int = Field(
        default=1024,
        ge=1,
        le=100000,
        description="Default number of shots for experiments",
    )
    default_phase_points: int = Field(
        default=20, ge=1, le=1000, description="Default number of phase points"
    )
    default_submit_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="Default interval between submissions in seconds",
    )
    default_wait_minutes: int = Field(
        default=30, ge=1, le=1440, description="Default wait time in minutes"
    )
    default_devices: list[str] = Field(
        default=["qulacs"], description="Default list of quantum devices"
    )

    @field_validator("default_devices")
    @classmethod
    def validate_devices(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one device must be specified")
        return v


class TranspilerOptions(BaseConfigModel):
    basis_gates: list[str] = Field(
        default=["sx", "x", "rz", "cx"], description="Basis gates for transpilation"
    )
    optimization_level: int = Field(
        default=2, ge=0, le=3, description="Optimization level for transpilation"
    )
    routing_method: str = Field(
        default="sabre", description="Routing method for transpilation"
    )

    @field_validator("basis_gates")
    @classmethod
    def validate_basis_gates(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one basis gate must be specified")
        return v

    @field_validator("routing_method")
    @classmethod
    def validate_routing_method(cls, v: str) -> str:
        allowed_methods = ["sabre", "basic", "lookahead", "stochastic"]
        if v not in allowed_methods:
            raise ValueError(f"Routing method must be one of {allowed_methods}")
        return v


class MitigationOptions(BaseConfigModel):
    ro_error_mitigation: str | None = Field(
        default="pseudo_inverse", description="Readout error mitigation method"
    )

    @field_validator("ro_error_mitigation")
    @classmethod
    def validate_mitigation_method(cls, v: str | None) -> str | None:
        if v is not None:
            allowed_methods = ["pseudo_inverse", "least_squares", None]
            if v not in allowed_methods:
                raise ValueError(f"Mitigation method must be one of {allowed_methods}")
        return v


class OQTOPUSSettings(BaseConfigModel):
    transpiler_options: TranspilerOptions = Field(
        default_factory=TranspilerOptions, description="Transpiler configuration"
    )
    mitigation_options: MitigationOptions = Field(
        default_factory=MitigationOptions, description="Error mitigation configuration"
    )


class WorkspaceInfo(BaseConfigModel):
    description: str = Field(description="Workspace description")
    version: str = Field(description="Workspace version")
    library_version: str = Field(description="Library version")


class DefaultConfig(BaseConfigModel):
    experiment_settings: ExperimentSettings = Field(
        default_factory=ExperimentSettings, description="General experiment settings"
    )
    oqtopus_settings: OQTOPUSSettings = Field(
        default_factory=OQTOPUSSettings, description="OQTOPUS platform settings"
    )
    chsh_parameters: CHSHParameters | None = Field(
        default_factory=CHSHParameters, description="CHSH experiment parameters"
    )
    workspace_info: WorkspaceInfo = Field(description="Workspace information")

    @classmethod
    def load_from_json(cls, file_path: str) -> "DefaultConfig":
        import json

        with open(file_path) as f:
            data = json.load(f)
        return cls(**data)

    def save_to_json(self, file_path: str) -> None:
        import json

        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)
