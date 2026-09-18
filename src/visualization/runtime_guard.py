"""Runtime memory guard for long visualization jobs."""

import psutil


def require_memory_margin(min_free_ratio: float = 0.20) -> float:
    """Raise before work if available system RAM is below the safety margin."""
    memory = psutil.virtual_memory()
    free_ratio = memory.available / memory.total
    if free_ratio < min_free_ratio:
        raise RuntimeError(
            f"insufficient RAM margin: {free_ratio:.1%} available, "
            f"minimum is {min_free_ratio:.1%}"
        )
    return free_ratio
