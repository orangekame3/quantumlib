from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        frozen=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfigModel":
        return cls(**data)


class BaseExperimentModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        use_enum_values=True,
    )


class BaseResultModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
