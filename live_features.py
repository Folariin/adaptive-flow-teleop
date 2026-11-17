"""Read active CSVs and calculate independent rolling baseline comparisons."""

import csv
import io
import math
import time
from bisect import bisect_left, bisect_right
from pathlib import Path

import features
import gsr_features
from experiment_session import METRICS, Baseline, deviations
from fuse_features import (
    MAX_XBOX_GAP_SECONDS, STEP_SECONDS, WINDOW_SECONDS,
    coverage_slice, gsr_bad_gaps,
)
from gsr_logger import MAX_ELAPSED_DISAGREEMENT_SECONDS
from recording_clock import verify_shared_clock


STALE_SECONDS = 2.0
JOYSTICK_METRICS = METRICS[:-1]
GSR_METRIC = METRICS[-1]


class RollingStream:
    def __init__(self, is_gsr: bool, window_origin: float | None = None):
        self.is_gsr = is_gsr
        self.samples = []
        self.timestamps = []
        self.first = None
        self.next_end = None
        self.latest = None
        self.window_origin = window_origin
        self.last_window = None

    def append(self, samples):
        for sample in samples:
            stamp = sample.pc_seconds if self.is_gsr else sample.timestamp
            if self.timestamps and stamp <= self.timestamps[-1]:
                raise ValueError("PC timestamps are not strictly increasing.")
            if self.first is None:
                self.first = sample
                origin = stamp if self.window_origin is None else self.window_origin
                self.next_end = origin + WINDOW_SECONDS
            if self.is_gsr:
                pc_elapsed = stamp - self.first.pc_seconds
                arduino_elapsed = (sample.arduino_millis - self.first.arduino_millis) / 1000
                if abs(pc_elapsed - arduino_elapsed) > MAX_ELAPSED_DISAGREEMENT_SECONDS:
                    raise ValueError("GSR acquisition and reception clocks disagree by more than 1 second.")
            self.samples.append(sample)
            self.timestamps.append(stamp)

    def update(self, now: float, *, latest_window: bool = True):
        if not self.timestamps:
            return "Waiting for complete samples.", None
        latest = self.timestamps[-1]
        if latest > now + STALE_SECONDS or now - latest > STALE_SECONDS:
            return "Samples are stale or the PC clock is incompatible.", None
        if latest < self.next_end:
            if self.latest is not None:
                return self.latest
            remaining = self.next_end - latest
            return f"Waiting for the first 30-second window ({remaining:.1f} s remaining).", None

        end = self.next_end
        if latest_window:
            end += STEP_SECONDS * math.floor((latest - end) / STEP_SECONDS)
        start = end - WINDOW_SECONDS
        self.next_end = end + STEP_SECONDS
        # Retain the sample before the left boundary to check gaps across it.
        left = max(0, bisect_left(self.timestamps, start) - 1)
        del self.timestamps[:left]
        del self.samples[:left]
        if self.is_gsr:
            gaps = gsr_bad_gaps(self.samples)
        else:
            gaps = [
                i for i, (a, b) in enumerate(zip(self.timestamps, self.timestamps[1:]), 1)
                if b - a > MAX_XBOX_GAP_SECONDS
            ]
        selected = coverage_slice(self.timestamps, gaps, start, end)
        self.last_window = (
            start, end, self.samples[bisect_left(self.timestamps, start):bisect_right(self.timestamps, end)],
        )
        if selected is None:
            self.latest = (f"Missing or discontinuous samples in window ending at PC {end:.3f} s.", None)
        else:
            window = self.samples[selected]
            if self.is_gsr:
                values = {GSR_METRIC: gsr_features.mean_raw(window)}
            else:
                result = features.calculate_features(window)
                values = {
                    "movement_variability": result.movement_variability,
                    "reversals": result.reversals,
                    "reversals_per_active_minute": result.reversals_per_active_minute,
                    "input_roughness_per_s": result.input_roughness,
                    "idle_seconds": result.idle_seconds,
                    "idle_percent": result.idle_percent,
                }
            self.latest = (f"Coverage OK · PC {start:.3f} to {end:.3f} s · {len(window)} samples", values)
        return self.latest


class CsvTail:
    def __init__(self, path: Path, is_gsr: bool):
        self.path = path
        self.offset = 0
        self.header = None
        self.stream = RollingStream(is_gsr)
        self.error = None

    def update(self):
        if self.error:
            return self.error, None
        try:
            with self.path.open("rb") as file:
                if file.seek(0, 2) < self.offset:
                    raise ValueError("Recording was truncated.")
                file.seek(self.offset)
                chunk = file.read(256 * 1024)
            complete = chunk.rfind(b"\n") + 1
            if complete:
                lines = chunk[:complete].decode("utf-8-sig")
                if self.header is None:
                    header, _, lines = lines.partition("\n")
                    self.header = header + "\n"
                if lines.strip():
                    data = io.StringIO(self.header + lines)
                    samples = (
                        gsr_features.read_samples(data, self.path) if self.stream.is_gsr
                        else features.read_samples(data)
                    )
                    self.stream.append(samples)
                self.offset += complete
            elif len(chunk) == 256 * 1024:
                raise ValueError("Recording contains an oversized CSV row.")
            return self.stream.update(time.perf_counter())
        except (OSError, ValueError, csv.Error) as error:
            self.error = f"Invalid live recording: {error}"
            return self.error, None


def format_comparison(status, values, baseline: Baseline) -> str:
    if values is None:
        return status

    def number(value):
        return "—" if value is None else f"{value:.4g}"

    lines = [status, f"{'Feature':<27} {'Current':>9} {'Base mean':>9} {'Change':>9} {'z':>9} {'Robust z':>9}"]
    for metric, value in values.items():
        summary = baseline.metrics[metric]
        difference, normalized, robust = deviations(value, summary)
        fields = (value, summary.mean, difference, normalized, robust)
        lines.append(f"{metric:<27} " + " ".join(f"{number(item):>9}" for item in fields))
    return "\n".join(lines)


def monitor_task(directory: Path, baseline: Baseline, stop, publish) -> None:
    tails = {}
    verified = None
    previous = None
    while not stop.is_set():
        results = {}
        gsr_paths = sorted(directory.glob("gsr_*.csv"))
        for name, paths in (
            ("gsr", gsr_paths),
            ("xbox", sorted((directory / "xbox_live").glob("controller_inputs_*.csv"))),
        ):
            if len(paths) != 1:
                results[name] = "Waiting for recording." if not paths else "Multiple recordings found in this task."
                continue
            path = paths[0]
            if name == "xbox":
                if len(gsr_paths) != 1:
                    results[name] = "Waiting for the GSR clock sidecar to verify synchronization."
                    continue
                pair = (path, gsr_paths[0])
                if pair != verified:
                    try:
                        verify_shared_clock(*pair)
                    except (OSError, ValueError) as error:
                        results[name] = f"Clock verification pending or failed: {error}"
                        continue
                    verified = pair
            if name not in tails or tails[name].path != path:
                tails[name] = CsvTail(path, name == "gsr")
            results[name] = format_comparison(*tails[name].update(), baseline)
        if results != previous:
            publish(results)
            previous = results
        stop.wait(0.25)
