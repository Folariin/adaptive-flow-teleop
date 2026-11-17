"""Record evidence that two Windows perf_counter streams share a clock."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple


CLOCK_SOURCE = "windows-cpython-query-performance-counter"
MAX_ORIGIN_DIFFERENCE_SECONDS = 1.0


class ClockMetadata(NamedTuple):
    host: str
    origin_unix_s: float


def metadata_path(recording: Path) -> Path:
    return recording.with_suffix(".clock.json")


def write_clock_metadata(recording: Path) -> None:
    """Write one sidecar. Clock metadata failures must not interrupt capture."""
    try:
        if sys.platform != "win32" or sys.implementation.name != "cpython" or sys.version_info < (3, 10):
            raise ValueError("clock verification requires Windows CPython 3.10 or later")
        if "QueryPerformanceCounter" not in time.get_clock_info("perf_counter").implementation:
            raise ValueError("unexpected perf_counter implementation")
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography",
            access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            machine_id, _ = winreg.QueryValueEx(key, "MachineGuid")
        host = hashlib.sha256(machine_id.encode("utf-8")).hexdigest()
        before = time.perf_counter()
        wall_time = time.time()
        after = time.perf_counter()
        if after - before > 0.010:
            raise ValueError("could not get a close wall-clock and PC counter reading")
        origin = wall_time - (before + after) / 2
        with metadata_path(recording).open("x", encoding="utf-8") as stream:
            json.dump({"source": CLOCK_SOURCE, "host": host, "origin_unix_s": origin}, stream)
    except (OSError, ValueError) as error:
        print(f"Warning: could not record clock metadata for {recording}: {error}", file=sys.stderr)


def read_clock_metadata(recording: Path, sidecar: Path | None = None) -> ClockMetadata | None:
    path = sidecar if sidecar is not None else metadata_path(recording)
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except FileNotFoundError:
        if sidecar is not None:
            raise
        return None
    except (ValueError, OSError) as error:
        raise ValueError(f"Could not read clock metadata in {path}: {error}") from error
    if (
        not isinstance(data, dict)
        or data.get("source") != CLOCK_SOURCE
        or not isinstance(data.get("host"), str)
        or not data["host"]
        or type(data.get("origin_unix_s")) not in (int, float)
        or not math.isfinite(data["origin_unix_s"])
    ):
        raise ValueError(f"Unsupported or invalid clock metadata in {path}.")
    return ClockMetadata(data["host"], data["origin_unix_s"])


def verify_shared_clock(
    xbox: Path, gsr: Path, confirm_shared_clock: bool = False, *,
    xbox_sidecar: Path | None = None, gsr_sidecar: Path | None = None,
) -> None:
    xbox_clock = read_clock_metadata(xbox, xbox_sidecar) if xbox_sidecar is not None else read_clock_metadata(xbox)
    gsr_clock = read_clock_metadata(gsr, gsr_sidecar) if gsr_sidecar is not None else read_clock_metadata(gsr)
    if xbox_clock is None or gsr_clock is None:
        if not confirm_shared_clock:
            raise ValueError(
                "Cannot verify a shared PC clock from the CSV timestamps alone. "
                "Clock sidecar missing. Use --confirm-shared-clock only if you verified "
                "both loggers ran on the same Windows boot with CPython 3.10 or later."
            )
        print("Warning: shared PC clock confirmed by the operator because clock metadata is missing.", file=sys.stderr)
        return
    if xbox_clock.host != gsr_clock.host:
        raise ValueError("Clock metadata identifies different Windows machines.")
    if abs(xbox_clock.origin_unix_s - gsr_clock.origin_unix_s) > MAX_ORIGIN_DIFFERENCE_SECONDS:
        raise ValueError("PC clock origins disagree by more than 1 second. Check for a reboot or wall-clock change.")
