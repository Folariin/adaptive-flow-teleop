"""Experimental, baseline-relative interpretation of GSR window measurements."""

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from experiment_session import Summary, deviations
from fuse_features import STEP_SECONDS, WINDOW_SECONDS
from gsr_diagnostics import Quality, Recording, inspect_signal
from gsr_logger import MAX_ELAPSED_DISAGREEMENT_SECONDS


class State(str, Enum):
    HIGHER = "higher_arousal_candidate"
    LOWER = "lower_arousal_candidate"
    BASELINE = "baseline_like"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Rules:
    direction: str = "unknown"
    direction_evidence: str = ""
    baseline_band_abs_z: float | None = None
    candidate_abs_z: float | None = None
    max_baseline_abs_drift_raw_per_s: float | None = None
    min_coverage_percent: float | None = None
    consecutive_windows: int = 1

    def __post_init__(self):
        if self.direction not in ("unknown", "higher_counts", "lower_counts"):
            raise ValueError("direction must be unknown, higher_counts, or lower_counts.")
        if not isinstance(self.direction_evidence, str):
            raise ValueError("direction_evidence must be text.")
        if type(self.consecutive_windows) is not int or self.consecutive_windows < 1:
            raise ValueError("consecutive_windows must be a positive integer.")
        for name in ("baseline_band_abs_z", "candidate_abs_z", "max_baseline_abs_drift_raw_per_s", "min_coverage_percent"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if self.candidate_abs_z is not None and self.candidate_abs_z <= (self.baseline_band_abs_z or 0):
            raise ValueError("candidate_abs_z must exceed the baseline band.")
        if self.min_coverage_percent is not None and not 0 < self.min_coverage_percent <= 100:
            raise ValueError("min_coverage_percent must be greater than 0 and at most 100.")


def load_rules(path: Path) -> Rules:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return Rules(**json.load(stream))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid GSR rules in {path}: {error}") from error


def quality_flags(quality: Quality) -> list[str]:
    flags = []
    if quality.invalid_rows:
        flags.append("invalid_rows")
    if quality.suspect_interval_count:
        flags.append("suspect_intervals")
    if quality.rail_samples:
        flags.append("adc_rail")
    if quality.max_elapsed_disagreement_s > MAX_ELAPSED_DISAGREEMENT_SECONDS:
        flags.append("clock_disagreement")
    return flags


class Interpreter:
    def __init__(self, baseline: Summary, baseline_quality: Quality | None, rules: Rules):
        self.baseline = baseline
        self.baseline_quality = baseline_quality
        self.rules = rules
        self.previous_end = None
        self.previous_candidate = None
        self.streak = 0

    def evaluate(self, start, end, current, samples, timing_status):
        rules = self.rules
        difference, z_score, robust_z = deviations(current, self.baseline)
        if z_score is not None and not math.isfinite(z_score):
            z_score = None
        if robust_z is not None and not math.isfinite(robust_z):
            robust_z = None
        quality, _ = inspect_signal(Recording(samples, 0, [], 0)) if samples else (None, [])
        coverage = (
            min(100.0, quality.covered_percent * quality.duration_s / WINDOW_SECONDS)
            if quality and quality.covered_percent is not None else None
        )
        reasons = []
        flags = quality_flags(quality) if quality else ["missing_samples"]
        if current is None:
            flags.append("replay_window_unavailable")
        elif not math.isfinite(current) or not 0 <= current <= 1023:
            flags.append("invalid_adc_mean")
            current, difference, z_score, robust_z = None, None, None, None
        if any(not 0 <= sample.raw <= 1023 for sample in samples):
            flags.append("invalid_adc_samples")
        if flags:
            reasons.append("Window quality: " + ", ".join(flags))
        baseline = self.baseline_quality
        if baseline is None:
            reasons.append("Raw baseline diagnostics are unavailable.")
        else:
            baseline_flags = quality_flags(baseline)
            if baseline_flags:
                reasons.append("Baseline quality: " + ", ".join(baseline_flags))
            if baseline.duration_s < WINDOW_SECONDS or baseline.drift_raw_per_s is None:
                reasons.append("Baseline is too short to assess stability.")
            elif rules.max_baseline_abs_drift_raw_per_s is None:
                reasons.append(f"Baseline drift is {baseline.drift_raw_per_s:+.4g} counts/s. No stability limit is configured.")
            elif abs(baseline.drift_raw_per_s) > rules.max_baseline_abs_drift_raw_per_s:
                reasons.append("Baseline drift exceeds the configured stability limit.")
            if rules.min_coverage_percent is not None and (
                baseline.covered_percent is None or baseline.covered_percent < rules.min_coverage_percent
            ):
                reasons.append("Baseline coverage is below the configured minimum.")
        if rules.direction == "unknown" or not rules.direction_evidence.strip():
            reasons.append("Verified sensor-response direction and calibration evidence are missing.")
        if rules.min_coverage_percent is None:
            reasons.append("No minimum window coverage is configured.")
        elif coverage is None or coverage < rules.min_coverage_percent:
            reasons.append("Window coverage is below the configured minimum.")
        if rules.baseline_band_abs_z is None or rules.candidate_abs_z is None:
            reasons.append("No complete experimental deviation rule is configured.")
        if z_score is None or not math.isfinite(z_score):
            reasons.append("Baseline-normalized deviation is unavailable.")

        state = State.UNCERTAIN
        candidate = None
        if not reasons:
            if abs(z_score) <= rules.baseline_band_abs_z:
                state = State.BASELINE
            elif abs(z_score) >= rules.candidate_abs_z:
                higher = (z_score > 0) == (rules.direction == "higher_counts")
                candidate = State.HIGHER if higher else State.LOWER
            else:
                reasons.append("Deviation lies between the configured baseline band and candidate threshold.")
        consecutive = self.previous_end is not None and math.isclose(
            end - self.previous_end, STEP_SECONDS, rel_tol=0, abs_tol=1e-6,
        )
        self.streak = (self.streak + 1 if consecutive and candidate == self.previous_candidate else 1) if candidate else 0
        self.previous_end, self.previous_candidate = end, candidate
        if candidate:
            if self.streak >= rules.consecutive_windows:
                state = candidate
            else:
                reasons.append(f"Sustained deviation pending ({self.streak}/{rules.consecutive_windows} windows).")
        return {
            "type": "experimental_gsr_window",
            "start_pc_s": start, "end_pc_s": end, "state": state.value,
            "raw_mean_adc": current, "baseline_mean_adc": self.baseline.mean,
            "difference_adc": difference, "z_score": z_score, "robust_z_score": robust_z,
            "coverage_percent": coverage, "timing_status": timing_status, "quality_flags": flags,
            "window_quality": asdict(quality) if quality else None,
            "baseline_summary": asdict(self.baseline),
            "baseline_quality": asdict(baseline) if baseline else None,
            "rules": asdict(rules), "consecutive_candidate_windows": self.streak,
            "reasons": reasons,
        }
