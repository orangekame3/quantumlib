"""
OQTOPUS Conversion Utilities

Common utilities for converting between different data formats used by OQTOPUS
and quantum experiment systems.
"""



def convert_decimal_to_binary_counts(
    decimal_counts: dict[str, int],
    num_qubits: int = 1
) -> dict[str, int]:
    """
    Convert OQTOPUS decimal counts to binary format.

    This function consolidates the conversion logic that was duplicated across
    T1, Ramsey, T2 Echo, and CHSH experiments.

    Args:
        decimal_counts: Dictionary with decimal keys and count values
        num_qubits: Number of qubits (1 for T1/Ramsey/T2, 2 for CHSH, etc.)

    Returns:
        Dictionary with binary string keys and count values

    Examples:
        For 1 qubit:
        0 -> "0"  (|0⟩ state)
        1 -> "1"  (|1⟩ state)

        For 2 qubits:
        0 -> "00" (|00⟩ state)
        1 -> "01" (|01⟩ state)
        2 -> "10" (|10⟩ state)
        3 -> "11" (|11⟩ state)
    """
    binary_counts = {}

    for decimal_key, count in decimal_counts.items():
        # Handle both string and numeric keys
        if isinstance(decimal_key, str):
            try:
                decimal_value = int(decimal_key)
            except ValueError:
                # Already in binary format
                binary_counts[decimal_key] = count
                continue
        else:
            decimal_value = int(decimal_key)

        # Convert decimal to binary string with proper padding
        try:
            binary_key = format(decimal_value, f'0{num_qubits}b')

            # Validate the result is within expected range
            if decimal_value >= 2**num_qubits:
                print(
                    f"⚠️ Unexpected count key: {decimal_key} "
                    f"(decimal value: {decimal_value}) for {num_qubits} qubits"
                )
                continue

        except (ValueError, OverflowError) as e:
            print(
                f"⚠️ Error converting key {decimal_key}: {e}"
            )
            continue

        # Accumulate counts for the same binary key
        if binary_key in binary_counts:
            binary_counts[binary_key] += count
        else:
            binary_counts[binary_key] = count

    return binary_counts


def validate_oqtopus_result(result: dict) -> bool:
    """
    Validate that an OQTOPUS result has the expected structure.

    Args:
        result: Result dictionary from OQTOPUS

    Returns:
        True if result structure is valid, False otherwise
    """
    if not result or not isinstance(result, dict):
        return False

    # Check for required status field
    if "status" not in result:
        return False

    # Check if status indicates success
    success_statuses = ["completed", "succeeded", "success"]
    status = result.get("status", "").lower()

    return status in success_statuses


def extract_counts_from_oqtopus_result(result: dict) -> dict[str, int] | None:
    """
    Extract measurement counts from OQTOPUS result structure.

    This function handles the various ways counts can be stored in OQTOPUS results
    and was consolidated from multiple experiment implementations.

    Args:
        result: OQTOPUS result dictionary

    Returns:
        Dictionary of measurement counts or None if not found
    """
    if not validate_oqtopus_result(result):
        return None

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

    # Convert to regular dict if needed (from Counter, etc.)
    if counts:
        return dict(counts)

    return None


def calculate_shots_from_counts(counts: dict[str, int]) -> int:
    """
    Calculate total number of shots from measurement counts.

    Args:
        counts: Dictionary of measurement counts

    Returns:
        Total number of shots
    """
    if not counts:
        return 0
    return sum(counts.values())


def normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    """
    Normalize counts to probabilities.

    Args:
        counts: Dictionary of measurement counts

    Returns:
        Dictionary of measurement probabilities (sum = 1.0)
    """
    total_shots = calculate_shots_from_counts(counts)
    if total_shots == 0:
        return {}

    return {state: count / total_shots for state, count in counts.items()}
