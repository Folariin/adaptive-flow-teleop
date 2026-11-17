"""Inspect saved GSR recordings and compare raw ADC features with a baseline."""

import argparse
import csv
import math
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from fuse_features import MAX_GSR_GAP_SECONDS, MAX_GSR_INTERVAL_DISAGREEMENT_SECONDS
from gsr_features import ANALYSIS_DIR, Sample, Window, calculate_windows, mean_raw
from gsr_logger import COLUMNS, MAX_ELAPSED_DISAGREEMENT_SECONDS


@dataclass
class Recording:
    samples: list[Sample]
    invalid_rows: int
    invalid_times: list[float]
    invalid_timestamps: int


def read_diagnostic_recording(path: Path) -> Recording:
    recording = Recording([], 0, [], 0)
    previous = -math.inf
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        if not set(COLUMNS).issubset(rows.fieldnames or ()):
            raise ValueError(f"{path}: expected columns {', '.join(COLUMNS)}")
        for row in rows:
            try:
                pc = float(row[COLUMNS[0]])
                arduino = int(row[COLUMNS[1]])
                if not math.isfinite(pc) or pc <= previous or not 0 <= arduino <= 0xFFFFFFFF:
                    raise ValueError
            except (TypeError, ValueError):
                recording.invalid_rows += 1
                recording.invalid_timestamps += 1
                continue
            previous = pc
            try:
                raw = int(row[COLUMNS[2]])
                if not 0 <= raw <= 1023:
                    raise ValueError
            except (TypeError, ValueError):
                recording.invalid_rows += 1
                recording.invalid_times.append(pc)
                continue
            recording.samples.append(Sample(pc, arduino, raw))
    if not recording.samples or not math.isfinite(recording.samples[-1].pc_seconds - recording.samples[0].pc_seconds):
        raise ValueError(f"{path}: no usable recording")
    return recording


def slope(samples):
    if len(samples) < 2:
        return None
    origin = samples[0].pc_seconds
    return statistics.linear_regression(
        [sample.pc_seconds - origin for sample in samples], [sample.raw for sample in samples],
    ).slope


@dataclass
class Quality:
    duration_s: float
    valid_samples: int
    invalid_rows: int
    invalid_timestamps: int
    median_rate_hz: float | None
    covered_percent: float | None
    suspect_interval_count: int
    max_pc_gap_s: float | None
    pc_gap_count: int
    arduino_gap_count: int
    arduino_nonincreasing_count: int
    max_elapsed_disagreement_s: float
    rail_samples: int
    sd_raw: float
    drift_raw_per_s: float | None


def inspect_signal(recording: Recording):
    samples = recording.samples
    first = samples[0]
    duration = samples[-1].pc_seconds - first.pc_seconds
    intervals, gaps = [], []
    pc_gaps = arduino_gaps = nonincreasing = 0
    for before, after in zip(samples, samples[1:]):
        pc_gap = after.pc_seconds - before.pc_seconds
        arduino_gap = (after.arduino_millis - before.arduino_millis) / 1000
        pc_gaps += pc_gap > MAX_GSR_GAP_SECONDS
        arduino_gaps += arduino_gap > MAX_GSR_GAP_SECONDS
        nonincreasing += arduino_gap <= 0
        invalid_index = bisect_right(recording.invalid_times, before.pc_seconds)
        invalid_between = invalid_index < len(recording.invalid_times) and recording.invalid_times[invalid_index] < after.pc_seconds
        if (
            pc_gap > MAX_GSR_GAP_SECONDS or not 0 < arduino_gap <= MAX_GSR_GAP_SECONDS
            or abs(pc_gap - arduino_gap) > MAX_GSR_INTERVAL_DISAGREEMENT_SECONDS or invalid_between
        ):
            gaps.append((before.pc_seconds, after.pc_seconds))
        intervals.append(pc_gap)
    disagreement = max(abs(
        (s.pc_seconds - first.pc_seconds) - (s.arduino_millis - first.arduino_millis) / 1000
    ) for s in samples)
    gap_time = math.fsum(end - start for start, end in gaps)
    return Quality(
        duration, len(samples), recording.invalid_rows, recording.invalid_timestamps,
        1 / statistics.median(intervals) if intervals else None,
        100 * max(0, duration - gap_time) / duration if duration else None,
        len(gaps), max(intervals, default=None), pc_gaps, arduino_gaps, nonincreasing, disagreement,
        sum(s.raw in (0, 1023) for s in samples),
        statistics.pstdev(s.raw for s in samples), slope(samples),
    ), gaps


def window_features(recording, quality, gaps, reference):
    samples = recording.samples
    times = [s.pc_seconds for s in samples]
    origin = times[0]
    gap_ends = [end for _, end in gaps]
    for window in calculate_windows(samples, baseline_mean=reference):
        start, end = origin + window.start_s, origin + window.end_s
        selected = samples[bisect_left(times, start):bisect_right(times, end)]
        gap_index = bisect_right(gap_ends, start)
        flags = []
        if not selected or (gap_index < len(gaps) and gaps[gap_index][0] < end):
            flags.append("gap_or_missing_data")
        if any(s.raw in (0, 1023) for s in selected):
            flags.append("adc_rail")
        if quality.invalid_timestamps or quality.max_elapsed_disagreement_s > MAX_ELAPSED_DISAGREEMENT_SECONDS:
            flags.append("recording_clock_warning")
        yield {
            **window._asdict(),
            "trend_raw_per_s": slope(selected),
            "within_window_sd_raw": statistics.pstdev(s.raw for s in selected) if len(selected) >= 2 else None,
            "quality": "|".join(flags) or "no_flag",
        }


def save_plot(path, recording, windows, gaps, reference, drift):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    samples = recording.samples
    origin = samples[0].pc_seconds
    elapsed = [s.pc_seconds - origin for s in samples]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained")
    ax.scatter(elapsed, [s.raw for s in samples], s=5, alpha=0.55, label="Raw samples")
    if windows:
        complete = [row for row in windows if row["mean_gsr_raw"] is not None]
        ax.scatter(
            [(row["start_s"] + row["end_s"]) / 2 for row in complete],
            [row["mean_gsr_raw"] for row in complete], s=20, label="30 s mean at window midpoint",
        )
    ax.axhline(reference, color="#187b50", linestyle="--", label="Baseline recording mean")
    for index, (start, end) in enumerate(gaps):
        ax.axvspan(start - origin, end - origin, color="#b94c20", alpha=0.2,
                   label="Suspect interval" if index == 0 else None)
    if drift is not None:
        intercept = statistics.mean(s.raw for s in samples) - drift * statistics.mean(elapsed)
        ax.plot([0, elapsed[-1]], [intercept, intercept + drift * elapsed[-1]],
                color="#643b90", linestyle=":", label="Linear drift fit")
    ax.set(xlabel="PC elapsed time (s)", ylabel="Raw ADC counts", title="GSR recording diagnostics")
    ax.legend()
    with path.open("xb") as output:
        fig.savefig(output, format="png", dpi=160)
    plt.close(fig)


def print_quality(label, quality):
    print(f"\n{label}")
    for key, value in asdict(quality).items():
        display = f"{value:.5g}" if isinstance(value, float) else ("N/A" if value is None else value)
        print(f"  {key}: {display}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="GSR recording to inspect")
    parser.add_argument("--baseline", type=Path, required=True, help="participant's baseline GSR CSV")
    parser.add_argument("--plot", action="store_true", help="save PNG diagnostics for both recordings (Matplotlib)")
    args = parser.parse_args()
    try:
        baseline = read_diagnostic_recording(args.baseline)
        recording = read_diagnostic_recording(args.recording)
        baseline_quality, baseline_gaps = inspect_signal(baseline)
        quality, gaps = inspect_signal(recording)
        reference = mean_raw(baseline.samples)
        windows = list(window_features(recording, quality, gaps, reference))
        print_quality("Baseline quality", baseline_quality)
        print_quality("Recording quality", quality)
        print(f"\nBaseline reference: {reference:.4f} raw ADC counts")
        print(f"Complete windows: {len(windows)}; flagged: {sum(row['quality'] != 'no_flag' for row in windows)}")
        print("Transient response metrics: unavailable for uncalibrated ADC recordings without artifact validation.")
        if (baseline_gaps or baseline.invalid_rows or baseline_quality.rail_samples
                or baseline_quality.max_elapsed_disagreement_s > MAX_ELAPSED_DISAGREEMENT_SECONDS
                or baseline_quality.duration_s < 30):
            print("Warning: baseline has quality or duration limitations. Review before interpreting differences.")
        ANALYSIS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prefix = ANALYSIS_DIR / f"{args.recording.stem}_{stamp}_gsr_diagnostics"
        path = prefix.with_name(f"{prefix.name}.csv")
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                *Window._fields, "trend_raw_per_s", "within_window_sd_raw", "quality",
            ])
            writer.writeheader()
            writer.writerows(windows)
        print(f"Window features: {path}")
        if args.plot:
            for label, data, rows, breaks, report in (
                ("recording", recording, windows, gaps, quality),
                ("baseline", baseline, list(window_features(baseline, baseline_quality, baseline_gaps, reference)),
                 baseline_gaps, baseline_quality),
            ):
                chart = prefix.with_name(f"{prefix.name}_{label}.png")
                save_plot(chart, data, rows, breaks, reference, report.drift_raw_per_s)
                print(f"Diagnostic plot: {chart}")
    except ImportError:
        parser.exit(1, "Plotting requires Matplotlib. Install it with python -m pip install matplotlib.\n")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
