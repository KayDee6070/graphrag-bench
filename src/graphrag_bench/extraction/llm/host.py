"""Host RAM checks that refuse to start inference rather than fail hours later.

A threshold is an engineering headroom estimate, not a measured requirement and not
a reservation. Another process can take the memory immediately after the check.
"""

import math
from pathlib import Path

from graphrag_bench.extraction.llm.config import LLMError

GIB = 1024**3


def available_memory_bytes(path: Path = Path("/proc/meminfo")) -> int | None:
    """Read Linux's available RAM estimate; unknown is never treated as sufficient."""
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields and fields[0] == "MemAvailable:":
                if len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit():
                    return int(fields[1]) * 1024
                return None
    except (OSError, UnicodeError):
        pass
    return None


def memory_check(threshold_gib: float | None) -> dict[str, object]:
    """Describe the host memory decision without deciding it."""
    available = available_memory_bytes()
    return {
        "min_available_gib": threshold_gib,
        "available_bytes": available,
        "sufficient": None
        if threshold_gib is None
        else available is not None and available >= threshold_gib * GIB,
    }


def require_memory(threshold_gib: float | None) -> dict[str, object]:
    """Raise before a model is constructed when available RAM is insufficient or unknown."""
    check = memory_check(threshold_gib)
    if threshold_gib is None:
        return check
    required = f"{threshold_gib:g}"
    if check["available_bytes"] is None:
        raise LLMError(f"cannot read available RAM, so the {required} GiB requirement is unmet")
    if not check["sufficient"]:
        gib = check["available_bytes"] / GIB
        raise LLMError(f"available RAM {gib:.2f} GiB is below the required {required} GiB")
    return check


def positive_gib(value: str) -> float:
    """Parse a finite positive GiB threshold; argparse reports the message."""
    try:
        gib = float(value)
    except ValueError as error:
        raise ValueError("RAM threshold must be a finite positive GiB value") from error
    if not math.isfinite(gib) or gib <= 0:
        raise ValueError("RAM threshold must be a finite positive GiB value")
    return gib
