from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from .base import BaseResultModel


class TaskMetadata(BaseResultModel):
    id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task execution status")
    created_at: datetime = Field(description="Task creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Task last update timestamp"
    )

    @field_validator("id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v or len(v) < 5:
            raise ValueError("Task ID must be at least 5 characters long")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed_statuses = [
            "pending",
            "running",
            "completed",
            "succeeded",
            "success",
            "failed",
            "error",
            "cancelled",
        ]
        if v.lower() not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}")
        return v.lower()


class CircuitResult(BaseResultModel):
    counts: dict[str, int] = Field(description="Measurement counts for quantum states")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional circuit metadata"
    )

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("Counts dictionary cannot be empty")

        for state, count in v.items():
            if count < 0:
                raise ValueError(f"Count for state {state} cannot be negative")
            if not isinstance(state, str):
                raise ValueError("State keys must be strings")

        return v

    @property
    def total_shots(self) -> int:
        return sum(self.counts.values())

    @property
    def probabilities(self) -> dict[str, float]:
        total = self.total_shots
        if total == 0:
            return {}
        return {state: count / total for state, count in self.counts.items()}


class OQTOPUSJobInfo(BaseResultModel):
    result: dict[str, Any] | None = Field(
        default=None, description="Job execution result"
    )
    job_info: dict[str, Any] | None = Field(
        default=None, description="Nested job information"
    )


class ExperimentResult(BaseResultModel):
    task_id: str = Field(description="Associated task identifier")
    circuit_results: list[CircuitResult] = Field(
        description="Results from circuit executions"
    )
    metadata: TaskMetadata = Field(description="Task execution metadata")
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Raw experimental data"
    )
    job_info: OQTOPUSJobInfo | None = Field(
        default=None, description="OQTOPUS job information"
    )

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v: str) -> str:
        if not v or len(v) < 5:
            raise ValueError("Invalid task ID format")
        return v

    @field_validator("circuit_results")
    @classmethod
    def validate_circuit_results(cls, v: list[CircuitResult]) -> list[CircuitResult]:
        if not v:
            raise ValueError("At least one circuit result must be provided")
        return v

    @property
    def total_shots(self) -> int:
        return sum(result.total_shots for result in self.circuit_results)

    @property
    def success_status(self) -> bool:
        success_statuses = ["completed", "succeeded", "success"]
        return self.metadata.status in success_statuses

    @classmethod
    def from_oqtopus_result(
        cls, result: dict[str, Any], task_id: str
    ) -> "ExperimentResult":
        if not cls._validate_oqtopus_structure(result):
            raise ValueError("Invalid OQTOPUS result structure")

        # Extract counts
        counts = cls._extract_counts_from_oqtopus(result)
        if not counts:
            raise ValueError("No measurement counts found in OQTOPUS result")

        # Create circuit result
        circuit_result = CircuitResult(counts=counts)

        # Create metadata
        metadata = TaskMetadata(
            id=task_id,
            status=result.get("status", "unknown"),
            created_at=datetime.now(),
        )

        # Create job info if present
        job_info = None
        if "job_info" in result:
            job_info = OQTOPUSJobInfo(
                result=result.get("job_info", {}).get("result"),
                job_info=result.get("job_info", {}).get("job_info"),
            )

        return cls(
            task_id=task_id,
            circuit_results=[circuit_result],
            metadata=metadata,
            raw_data=result,
            job_info=job_info,
        )

    @staticmethod
    def _validate_oqtopus_structure(result: dict[str, Any]) -> bool:
        if not result or not isinstance(result, dict):
            return False

        if "status" not in result:
            return False

        success_statuses = ["completed", "succeeded", "success"]
        status = result.get("status", "").lower()
        return status in success_statuses

    @staticmethod
    def _extract_counts_from_oqtopus(result: dict[str, Any]) -> dict[str, int] | None:
        counts = None

        # Method 1: Direct counts in result
        if "counts" in result:
            counts = result["counts"]

        # Method 2: Get from job_info.result.sampling structure
        elif "job_info" in result:
            job_info = result.get("job_info", {})
            if isinstance(job_info, dict):
                sampling_result = job_info.get("result", {}).get("sampling", {})
                if sampling_result:
                    counts = sampling_result.get("counts", {})

            # Method 3: Nested job_info structure
            if not counts and isinstance(job_info, dict) and "job_info" in job_info:
                inner_job_info = job_info["job_info"]
                if isinstance(inner_job_info, dict):
                    result_data = inner_job_info.get("result", {})
                    if "sampling" in result_data:
                        counts = result_data["sampling"].get("counts", {})
                    elif "counts" in result_data:
                        counts = result_data["counts"]

        # Convert to proper format
        if counts:
            # Convert keys to strings and ensure int values
            return {str(k): int(v) for k, v in counts.items()}

        return None
