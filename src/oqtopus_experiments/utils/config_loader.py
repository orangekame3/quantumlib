import json
from pathlib import Path
from typing import Any

from ..models.config import DefaultConfig


class ConfigLoader:
    @staticmethod
    def load_default_config(config_path: str | Path | None = None) -> DefaultConfig:
        if config_path is None:
            # Default path
            config_path = (
                Path(__file__).parent.parent.parent.parent
                / "workspace"
                / "configs"
                / "default_experiment_config.json"
            )

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        return DefaultConfig(**data)

    @staticmethod
    def load_legacy_config_as_dict(config_path: str | Path) -> dict[str, Any]:
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def migrate_config_to_pydantic(
        legacy_config_path: str | Path, output_path: str | Path | None = None
    ) -> DefaultConfig:
        legacy_data = ConfigLoader.load_legacy_config_as_dict(legacy_config_path)
        pydantic_config = DefaultConfig(**legacy_data)

        if output_path:
            pydantic_config.save_to_json(str(output_path))

        return pydantic_config
