"""Summarize raw GSR ADC readings in complete sliding windows."""

import argparse
import csv
import math
import statistics
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from gsr_logger import COLUMNS, TimingChecks


ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis"
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_STEP_SECONDS = 5.0
MAX_PC_GAP_SECONDS = 0.5


class Sample(NamedTuple):
    pc_seconds: float
    arduino_millis: int
    raw: int


class Window(NamedTuple):
    start_s: float
    end_s: float
    sample_count: int
    mean_gsr_raw: float | None
    baseline_mean_gsr_raw: float | None
    difference_from_baseline_raw: float | None


class RecordingTimingChecks(TimingChecks):
    def __init__(self, path: Path, first: Sample):
        super().__init__(first.pc_seconds, first.arduino_millis)
        self.path = path

    def warn(self, kind: str, message: str) -> None:
        super().warn(kind, f"{self.path}: {message}")


def read_recording(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return read_samples(stream, path)


def read_samples(stream, path: Path) -> list[Sample]:
    samples = []
    reader = csv.DictReader(stream, strict=True)
    if not reader.fieldnames:
        raise ValueError(f"{path}: recording is empty.")
    missing = [column for column in COLUMNS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

    for row in reader:
        location = f"{path}, line {reader.line_num}"
        try:
            sample = Sample(float(row[COLUMNS[0]]), int(row[COLUMNS[1]]), int(row[COLUMNS[2]]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{location}: invalid timestamp or raw ADC value.") from error
        if not math.isfinite(sample.pc_seconds) or not 0 <= sample.arduino_millis <= 0xFFFFFFFF:
            raise ValueError(f"{location}: invalid timestamp.")
        if not 0 <= sample.raw <= 1023:
            raise ValueError(f"{location}: raw ADC value must be between 0 and 1023.")
        if samples and sample.pc_seconds <= samples[-1].pc_seconds:
            raise ValueError(f"{location}: pc_perf_counter_s must be strictly increasing.")
        samples.append(sample)

    if not samples:
        raise ValueError(f"{path}: recording contains no samples.")
    if not math.isfinite(samples[-1].pc_seconds - samples[0].pc_seconds):
        raise ValueError(f"{path}: elapsed PC time must be finite.")
    return samples


def warn_about_timing(path: Path, samples: Sequence[Sample]) -> None:
    checks = RecordingTimingChecks(path, samples[0])
    for previous, current in zip(samples, samples[1:]):
        pc_gap = current.pc_seconds - previous.pc_seconds
        if pc_gap > MAX_PC_GAP_SECONDS:
            offset = current.pc_seconds - samples[0].pc_seconds
            checks.warn("pc_gap", f"PC sampling gap of {pc_gap:.3f} s ending at {offset:.3f} s.")
        checks.observe(current.pc_seconds, current.arduino_millis)
    if checks.warned:
        print(f"Warning: {path}: timing issues may make analysis unreliable.", file=sys.stderr)


def mean_raw(samples: Sequence[Sample]) -> float:
    return statistics.mean(sample.raw for sample in samples)


def calculate_windows(
    samples: Sequence[Sample],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    baseline_mean: float | None = None,
) -> list[Window]:
    """Use both inclusive bounds on the PC clock and weight each sample equally."""
    for name, value in (("Window duration", window_seconds), ("Window step", step_seconds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")
    if not samples:
        return []

    timestamps = [sample.pc_seconds for sample in samples]
    origin = timestamps[0]
    windows = []
    while True:
        start = len(windows) * step_seconds
        end = start + window_seconds
        if origin + end > timestamps[-1]:
            return windows
        left = bisect_left(timestamps, origin + start)
        right = bisect_right(timestamps, origin + end)
        average = mean_raw(samples[left:right]) if right > left else None
        difference = average - baseline_mean if average is not None and baseline_mean is not None else None
        windows.append(Window(start, end, right - left, average, baseline_mean, difference))


def export_windows(recording: Path, windows: Sequence[Window]) -> Path:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = ANALYSIS_DIR / f"{recording.stem}_{timestamp}_gsr_windows.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(Window._fields)
        writer.writerows(windows)
    return path


def format_mean(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="CSV written by gsr_logger.py")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="window duration (default: 30)")
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS, help="window step (default: 5)")
    parser.add_argument("--baseline", type=Path, help="resting GSR CSV from the same logger")
    parser.add_argument("--export", action="store_true", help="save window results to analysis/")
    args = parser.parse_args(argv)
    try:
        samples = read_recording(args.recording)
        warn_about_timing(args.recording, samples)
        baseline_mean = None
        if args.baseline is not None:
            baseline = read_recording(args.baseline)
            warn_about_timing(args.baseline, baseline)
            baseline_mean = mean_raw(baseline)
        windows = calculate_windows(samples, args.window_seconds, args.step_seconds, baseline_mean)

        elapsed = samples[-1].pc_seconds - samples[0].pc_seconds
        print(f"Recording: {args.recording} ({len(samples)} samples over {elapsed:.4f} s)")
        if baseline_mean is not None:
            print(f"Resting baseline: {baseline_mean:.4f} raw ADC counts ({len(baseline)} samples)")
        if not windows:
            print("No complete windows. The recording is shorter than the requested window.")
        else:
            header = "Start (s) | End (s) | Samples | Mean GSR (raw ADC counts)"
            if baseline_mean is not None:
                header += " | Difference from baseline (counts)"
            print(header)
            for window in windows:
                line = f"{window.start_s:.4f} | {window.end_s:.4f} | {window.sample_count} | {format_mean(window.mean_gsr_raw)}"
                if baseline_mean is not None:
                    line += f" | {format_mean(window.difference_from_baseline_raw)}"
                print(line)
        if args.export:
            print(f"Window CSV: {export_windows(args.recording, windows)}")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
