"""Extract behavioural features from an Xbox controller recording offline."""

import argparse
import csv
import math
import statistics
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


AXIS_COLUMNS = ("lx_deadzone", "ly_deadzone", "rx_deadzone", "ry_deadzone")
REQUIRED_COLUMNS = ("perf_counter_s", *AXIS_COLUMNS, "lt", "rt", "x_pressed", "rb_pressed")
REVERSAL_THRESHOLD = 0.05
MAX_NEUTRAL_SECONDS = 0.5
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_STEP_SECONDS = 5.0
WINDOW_COLUMNS = (
    "start_s", "end_s", "observed_duration_s", "movement_variability",
    "reversals", "reversals_per_active_minute", "input_roughness_per_s", "idle_percent",
)


@dataclass(frozen=True)
class Sample:
    timestamp: float
    axes: tuple[float, ...]
    idle: bool


@dataclass(frozen=True)
class Features:
    duration: float
    movement_variability: float
    reversals: int
    active_seconds: float
    reversals_per_active_minute: float | None
    input_roughness: float | None
    idle_seconds: float
    idle_percent: float | None


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    features: Features | None

    def row(self) -> tuple[float | int | None, ...]:
        result = self.features
        if result is None:
            return (self.start, self.end, None, None, None, None, None, None)
        return (
            self.start, self.end, result.duration, result.movement_variability,
            result.reversals, result.reversals_per_active_minute,
            result.input_roughness, result.idle_percent,
        )


def read_recording(path: Path) -> list[Sample]:
    """Read the required logger columns, rejecting malformed rows with their line number."""
    with path.open(newline="", encoding="utf-8-sig") as recording:
        return read_samples(recording)


def read_samples(recording) -> list[Sample]:
    samples = []
    reader = csv.DictReader(recording, strict=True)
    if not reader.fieldnames:
        raise ValueError("The recording is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for row in reader:
        values = {}
        for column in REQUIRED_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                raise ValueError(f"Line {reader.line_num}: invalid value for {column}.") from None
            if not math.isfinite(value):
                raise ValueError(f"Line {reader.line_num}: {column} must be finite.")
            values[column] = value

        axes = tuple(values[column] for column in AXIS_COLUMNS)
        idle = not any((*axes, values["lt"], values["rt"], values["x_pressed"], values["rb_pressed"]))
        samples.append(Sample(values["perf_counter_s"], axes, idle))
    return samples


def count_reversals(timestamps: Sequence[float], axis: Sequence[float]) -> int:
    """Count opposite signs with at most 0.5 s of intervening neutral input.

    Magnitudes below 0.05 are neutral. A neutral interval starts at the first
    neutral sample and ends at the next meaningful sample. A direct sign
    change has no neutral interval.
    """
    reversals = 0
    previous_sign = 0
    neutral_since = None
    for timestamp, value in zip(timestamps, axis):
        if abs(value) < REVERSAL_THRESHOLD:
            if neutral_since is None:
                neutral_since = timestamp
            continue

        sign = 1 if value > 0 else -1
        if previous_sign == -sign and (
            neutral_since is None or timestamp - neutral_since <= MAX_NEUTRAL_SECONDS
        ):
            reversals += 1
        previous_sign = sign
        neutral_since = None
    return reversals


def recording_timestamps(samples: Sequence[Sample]) -> list[float]:
    if not samples:
        raise ValueError("The recording contains no samples.")

    timestamps = [sample.timestamp for sample in samples]
    for previous, current in zip(timestamps, timestamps[1:]):
        if not current > previous:
            raise ValueError("perf_counter_s timestamps must be strictly increasing.")
    if not math.isfinite(timestamps[-1] - timestamps[0]):
        raise ValueError("The elapsed recording duration must be finite.")
    return timestamps


def calculate_features(samples: Sequence[Sample]) -> Features:
    """Analyze one period. Each interval belongs to its earlier sample.

    Active time is the sum of intervals whose earlier sample is non-idle.
    The last sample has no measured interval after it.
    """
    timestamps = recording_timestamps(samples)
    duration = timestamps[-1] - timestamps[0]

    axes = tuple(zip(*(sample.axes for sample in samples)))
    variability = statistics.mean(statistics.pstdev(axis) for axis in axes)
    reversals = sum(count_reversals(timestamps, axis) for axis in axes)
    total_change = math.fsum(
        abs(current - previous)
        for axis in axes
        for previous, current in zip(axis, axis[1:])
    )
    idle_seconds = 0.0
    active_seconds = 0.0
    for sample, next_timestamp in zip(samples, timestamps[1:]):
        interval = next_timestamp - sample.timestamp
        if sample.idle:
            idle_seconds += interval
        else:
            active_seconds += interval

    return Features(
        duration=duration,
        movement_variability=variability,
        reversals=reversals,
        active_seconds=active_seconds,
        reversals_per_active_minute=60 * reversals / active_seconds if active_seconds else None,
        input_roughness=total_change / (4 * duration) if duration else None,
        idle_seconds=idle_seconds,
        idle_percent=100 * idle_seconds / duration if duration else None,
    )


def calculate_windows(
    samples: Sequence[Sample],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
) -> list[Window]:
    """Analyze complete windows anchored at the first recorded timestamp.

    Both endpoints are inclusive. Features use only samples inside the window
    and the elapsed span between the first and last of those samples. No
    interval is extrapolated from the last selected sample to the window end.
    """
    for name, value in (("Window duration", window_seconds), ("Window step", step_seconds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    timestamps = recording_timestamps(samples)
    origin = timestamps[0]
    windows = []
    while True:
        start = len(windows) * step_seconds
        end = start + window_seconds
        if origin + end > timestamps[-1]:
            return windows
        left = bisect_left(timestamps, origin + start)
        right = bisect_right(timestamps, origin + end)
        selected = samples[left:right]
        windows.append(Window(start, end, calculate_features(selected) if selected else None))


def output_path(recording: Path, suffix: str) -> Path:
    folder = Path(__file__).resolve().parent / "analysis"
    folder.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return folder / f"{recording.stem}_{timestamp}_{suffix}"


def export_windows(recording: Path, windows: Sequence[Window]) -> Path:
    path = output_path(recording, "windows.csv")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(WINDOW_COLUMNS)
        writer.writerows(window.row() for window in windows)
    return path


def plot_axes(recording: Path, samples: Sequence[Sample]) -> Path:
    """Save a diagnostic SVG with four panels sharing the elapsed-time scale."""
    elapsed = [sample.timestamp - samples[0].timestamp for sample in samples]
    duration = elapsed[-1]
    left, width, amplitude = 80, 870, 55
    x_positions = [left + width * (time / duration if duration else 0) for time in elapsed]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 680" role="img">',
        '<title>Deadzone-adjusted joystick input over elapsed time</title>',
        '<rect width="980" height="680" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#182230">',
        '<text x="490" y="28" text-anchor="middle" font-size="19">Deadzone-adjusted joystick input</text>',
    ]
    for index, (label, values) in enumerate(zip(("LX", "LY", "RX", "RY"), zip(*(s.axes for s in samples)))):
        center = 110 + 145 * index
        svg.append(f'<text x="20" y="{center + 4}">{label}</text>')
        for level in (1, 0, -1):
            y = center - amplitude * level
            svg.append(f'<path d="M {left} {y} h {width}" stroke="#cdd5df"/>')
            svg.append(f'<text x="68" y="{y + 4}" text-anchor="end">{level}</text>')
        points = [(x, center - amplitude * value) for x, value in zip(x_positions, values)]
        if len(points) == 1:
            x, y = points[0]
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#245ba3"/>')
        else:
            coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="#245ba3"/>')
    for tick in range(6 if duration else 1):
        x = left + width * tick / 5
        seconds = duration * (tick / 5)
        svg.append(f'<text x="{x:.2f}" y="625" text-anchor="middle">{seconds:.3g}</text>')
    svg.append('<text x="515" y="655" text-anchor="middle">Elapsed time (s)</text></g></svg>')
    path = output_path(recording, "joysticks.svg")
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(svg))
    return path


def format_value(value: float | None, unit: str = "") -> str:
    return "N/A" if value is None else f"{value:.4f}{unit}"


def print_summary(features: Features) -> None:
    print(f"Recorded duration: {features.duration:.4f} s")
    print(f"Movement variability: {features.movement_variability:.4f}")
    print(f"Direction reversals: {features.reversals}")
    print(f"Active time (non-idle controller): {features.active_seconds:.4f} s")
    print(f"Reversals per active minute: {format_value(features.reversals_per_active_minute)}")
    print(f"Input roughness proxy: {format_value(features.input_roughness, ' /s')}")
    print(f"Idle time: {features.idle_seconds:.4f} s ({format_value(features.idle_percent, '%')})")


def print_windows(windows: Sequence[Window]) -> None:
    if not windows:
        print("No complete windows. The recording is shorter than the requested window.")
        return
    print("Start (s) | End (s) | Observed (s) | Variability | Reversals | Reversals/active min | Roughness (/s) | Idle (%)")
    for window in windows:
        start, end, observed, variability, reversals, rate, roughness, idle = window.row()
        print(
            f"{start:.4f} | {end:.4f} | {format_value(observed)} | {format_value(variability)} | "
            f"{reversals if reversals is not None else 'N/A'} | {format_value(rate)} | "
            f"{format_value(roughness)} | {format_value(idle)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="CSV written by xbox_controller.py")
    parser.add_argument("--sliding", action="store_true", help="analyze complete sliding windows")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="window duration (default: 30)")
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS, help="window step (default: 5)")
    parser.add_argument("--export", action="store_true", help="save sliding results as CSV in analysis/")
    parser.add_argument("--plot", action="store_true", help="save a joystick SVG in analysis/")
    args = parser.parse_args()
    if args.export and not args.sliding:
        parser.error("--export requires --sliding")

    try:
        samples = read_recording(args.recording)
        print(f"Recording: {args.recording}")
        print(f"Samples: {len(samples)}")
        if args.sliding:
            windows = calculate_windows(samples, args.window_seconds, args.step_seconds)
            print_windows(windows)
            if args.export:
                print(f"Window CSV: {export_windows(args.recording, windows)}")
        else:
            print_summary(calculate_features(samples))
        if args.plot:
            print(f"Joystick plot: {plot_axes(args.recording, samples)}")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error reading or analyzing {args.recording}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
