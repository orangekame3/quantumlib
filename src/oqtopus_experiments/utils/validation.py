from typing import Any

from ..models.config import DefaultConfig
from ..models.results import ExperimentResult


def validate_oqtopus_result_legacy(result: dict[str, Any]) -> bool:
    if not result or not isinstance(result, dict):
        return False

    if "status" not in result:
        return False

    success_statuses = ["completed", "succeeded", "success"]
    status = result.get("status", "").lower()
    return status in success_statuses


def validate_and_convert_oqtopus_result(
    result: dict[str, Any], task_id: str
) -> ExperimentResult | None:
    try:
        return ExperimentResult.from_oqtopus_result(result, task_id)
    except Exception as e:
        print(f"⚠️ Failed to convert OQTOPUS result to pydantic model: {e}")
        return None


def validate_config_migration(legacy_config_path: str) -> bool:
    try:
        DefaultConfig.load_from_json(legacy_config_path)
        return True
    except Exception as e:
        print(f"⚠️ Configuration validation failed: {e}")
        return False
