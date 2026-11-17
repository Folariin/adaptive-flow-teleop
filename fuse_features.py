"""Fuse Xbox and GSR features in shared PC-clock windows."""

import argparse
import csv
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import features
import gsr_features
from gsr_logger import MAX_ELAPSED_DISAGREEMENT_SECONDS
from recording_clock import verify_shared_clock


ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
WINDOW_SECONDS = 30
STEP_SECONDS = 5
MAX_XBOX_GAP_SECONDS = 0.090
MAX_GSR_GAP_SECONDS = 0.225
MAX_GSR_INTERVAL_DISAGREEMENT_SECONDS = 0.075


class FusedWindow(NamedTuple):
    start_pc_s: float
    end_pc_s: float
    xbox_sample_count: int
    gsr_sample_count: int
    xbox_observed_duration_s: float
    movement_variability: float
    reversals: int
    active_seconds: float
    reversals_per_active_minute: float | None
    input_roughness_per_s: float
    idle_seconds: float
    idle_percent: float
    mean_gsr_raw: float


def coverage_slice(
    timestamps: Sequence[float], bad_gap_ends: Sequence[int], start: float, end: float,
) -> slice | None:
    """Select samples inclusively if all intervals crossing the window are valid."""
    left = bisect_left(timestamps, start)
    right = bisect_right(timestamps, end)
    if right - left < 2:
        return None
    # A gap ending exactly at the start or starting exactly at the end is outside.
    first_gap = bisect_left(bad_gap_ends, bisect_right(timestamps, start))
    if first_gap < len(bad_gap_ends) and bad_gap_ends[first_gap] <= bisect_left(timestamps, end):
        return None
    return slice(left, right)


def gsr_bad_gaps(samples: Sequence[gsr_features.Sample]) -> list[int]:
    first = samples[0]
    bad_gaps = []
    for index, (previous, current) in enumerate(zip(samples, samples[1:]), start=1):
        pc_elapsed = current.pc_seconds - first.pc_seconds
        arduino_elapsed = (current.arduino_millis - first.arduino_millis) / 1000
        if abs(pc_elapsed - arduino_elapsed) > MAX_ELAPSED_DISAGREEMENT_SECONDS:
            raise ValueError(
                f"GSR clock timing is unreliable at PC time {current.pc_seconds:.6f} s "
                f"(Arduino elapsed {arduino_elapsed:.3f} s, PC elapsed {pc_elapsed:.3f} s)."
            )
        pc_gap = current.pc_seconds - previous.pc_seconds
        arduino_gap = (current.arduino_millis - previous.arduino_millis) / 1000
        if (
            pc_gap > MAX_GSR_GAP_SECONDS
            or not 0 < arduino_gap <= MAX_GSR_GAP_SECONDS
            or abs(pc_gap - arduino_gap) > MAX_GSR_INTERVAL_DISAGREEMENT_SECONDS
        ):
            bad_gaps.append(index)
    return bad_gaps


def fuse_recordings(
    xbox_samples: Sequence[features.Sample], gsr_samples: Sequence[gsr_features.Sample],
    warn: Callable[[str], None] | None = None,
) -> list[FusedWindow]:
    if warn is None:
        def warn(message: str) -> None:
            print(message, file=sys.stderr)
    xbox_times = features.recording_timestamps(xbox_samples)
    gsr_times = [sample.pc_seconds for sample in gsr_samples]
    origin = max(xbox_times[0], gsr_times[0])
    overlap_end = min(xbox_times[-1], gsr_times[-1])
    if overlap_end <= origin:
        raise ValueError("Recordings do not overlap on the absolute PC clock.")
    if origin + WINDOW_SECONDS > overlap_end:
        raise ValueError(f"Shared PC-time coverage is only {overlap_end - origin:.3f} s. At least 30 s is required.")

    xbox_gaps = [
        index for index, (a, b) in enumerate(zip(xbox_times, xbox_times[1:]), start=1)
        if b - a > MAX_XBOX_GAP_SECONDS
    ]
    gsr_gaps = gsr_bad_gaps(gsr_samples)
    if xbox_gaps or gsr_gaps:
        warn(
            f"Warning: found {len(xbox_gaps)} Xbox and {len(gsr_gaps)} GSR suspect sample intervals "
            "in the source recordings. Windows crossing those intervals are excluded."
        )

    windows = []
    skipped = 0
    candidate_count = int((overlap_end - origin - WINDOW_SECONDS) // STEP_SECONDS) + 1
    for index in range(candidate_count):
        start = origin + index * STEP_SECONDS
        end = start + WINDOW_SECONDS
        xbox_selection = coverage_slice(xbox_times, xbox_gaps, start, end)
        gsr_selection = coverage_slice(gsr_times, gsr_gaps, start, end)
        if xbox_selection is None or gsr_selection is None:
            skipped += 1
            continue
        xbox_window = xbox_samples[xbox_selection]
        gsr_window = gsr_samples[gsr_selection]
        result = features.calculate_features(xbox_window)
        windows.append(FusedWindow(
            start, end, len(xbox_window), len(gsr_window), result.duration,
            result.movement_variability, result.reversals, result.active_seconds,
            result.reversals_per_active_minute, result.input_roughness, result.idle_seconds,
            result.idle_percent, gsr_features.mean_raw(gsr_window),
        ))
    if skipped:
        warn(f"Warning: skipped {skipped} of {candidate_count} windows with unreliable sample coverage.")
    if not windows:
        raise ValueError("No complete window has valid coverage from both recordings.")
    return windows


def export_windows(windows: Sequence[FusedWindow]) -> Path:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = ANALYSIS_DIR / f"fused_features_{timestamp}.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FusedWindow._fields)
        writer.writerows(windows)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xbox", type=Path, help="controller_inputs_*.csv")
    parser.add_argument("gsr", type=Path, help="gsr_*.csv")
    parser.add_argument(
        "--confirm-shared-clock", action="store_true",
        help="confirm the same Windows boot and CPython 3.10+ for legacy files without clock sidecars",
    )
    args = parser.parse_args(argv)
    try:
        verify_shared_clock(args.xbox, args.gsr, args.confirm_shared_clock)
        xbox_samples = features.read_recording(args.xbox)
        gsr_samples = gsr_features.read_recording(args.gsr)
        windows = fuse_recordings(xbox_samples, gsr_samples)
        path = export_windows(windows)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Fusion failed: {error}\n")
    print(f"Exported {len(windows)} synchronized windows to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
