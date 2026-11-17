"""Detect behavioural changes and apply an optional experimental joystick rule."""

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiment_session import deviations
from fuse_features import MAX_XBOX_GAP_SECONDS, STEP_SECONDS, WINDOW_SECONDS
from live_features import JOYSTICK_METRICS


FAMILIES = {
    "movement_variability": ("movement_variability",),
    "reversals": ("reversals_per_active_minute", "reversals"),
    "input_roughness": ("input_roughness_per_s",),
    "idle": ("idle_percent", "idle_seconds"),
}


@dataclass(frozen=True)
class Rules:
    baseline_band_abs_z: float | None = None
    change_abs_z: float | None = None
    consecutive_windows: int = 1
    min_coverage_percent: float | None = None
    family_metrics: dict = field(default_factory=lambda: {family: metrics[0] for family, metrics in FAMILIES.items()})
    higher_pattern: dict = field(default_factory=dict)
    mapping_evidence: str = ""
    mapping_context: dict = field(default_factory=dict)
    conflict_policy: str = "require_full_pattern"

    def __post_init__(self):
        for name in ("baseline_band_abs_z", "change_abs_z", "min_coverage_percent"):
            value = getattr(self, name)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite nonnegative number.")
        if self.change_abs_z is not None and self.change_abs_z <= (self.baseline_band_abs_z or 0):
            raise ValueError("change_abs_z must exceed the baseline band.")
        if self.min_coverage_percent is not None and not 0 < self.min_coverage_percent <= 100:
            raise ValueError("min_coverage_percent must be greater than 0 and at most 100.")
        if type(self.consecutive_windows) is not int or self.consecutive_windows < 1:
            raise ValueError("consecutive_windows must be a positive integer.")
        if not isinstance(self.family_metrics, dict) or self.family_metrics.keys() != FAMILIES.keys() or any(
            metric not in FAMILIES[family] for family, metric in self.family_metrics.items()
        ):
            raise ValueError("Choose exactly one existing metric for each behavioural family.")
        if not isinstance(self.higher_pattern, dict) or any(
            family not in FAMILIES or direction not in ("increased", "decreased")
            for family, direction in self.higher_pattern.items()
        ):
            raise ValueError("higher_pattern must specify increased or decreased by behavioural family.")
        if self.conflict_policy not in ("require_full_pattern", "allow_baseline_like", "majority_direction"):
            raise ValueError("Unknown conflict_policy.")
        if not isinstance(self.mapping_evidence, str):
            raise ValueError("mapping_evidence must be text.")
        validate_context(self.mapping_context)


def validate_context(context):
    if not isinstance(context, dict) or any(
        key not in ("task_type", "difficulty") or not isinstance(value, str) or not value.strip()
        for key, value in context.items()
    ):
        raise ValueError("Task context must contain text values for task_type or difficulty.")


def load_configuration(path: Path):
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        context = data.pop("task_context", {})
        validate_context(context)
        return Rules(**data), context
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid joystick configuration in {path}: {error}") from error


class Interpreter:
    def __init__(self, baseline, rules: Rules):
        self.baseline = baseline
        self.rules = rules
        self.previous_end = None
        self.previous_context = None
        self.runs = {}
        self.active_pattern = None
        self.pattern_streak = 0

    def evaluate(self, start, end, values, samples, timing_status, context=None):
        rules = self.rules
        context = dict(context or {})
        validate_context(context)
        coverage = min(100.0, 100 * (samples[-1].timestamp - samples[0].timestamp) / WINDOW_SECONDS) if len(samples) >= 2 else None
        quality = []
        if values is None or coverage is None:
            quality.append("Missing, invalid, or discontinuous joystick samples.")
        if any(not math.isfinite(sample.timestamp) for sample in samples) or any(
            not 0 < after.timestamp - before.timestamp <= MAX_XBOX_GAP_SECONDS
            for before, after in zip(samples, samples[1:])
        ):
            quality.append("Invalid timestamps or joystick sampling gap.")
        if coverage is not None and (not math.isfinite(coverage) or coverage < 0):
            coverage = None
        if any(not math.isfinite(axis) or abs(axis) > 1 for sample in samples for axis in sample.axes):
            quality.append("Invalid processed joystick axis.")
        if rules.min_coverage_percent is None:
            quality.append("Minimum window coverage is unconfigured.")
        elif coverage is None or coverage < rules.min_coverage_percent:
            quality.append("Window coverage is below the configured minimum.")
        boundaries_set = rules.baseline_band_abs_z is not None and rules.change_abs_z is not None
        if self.previous_end is None or not math.isclose(
            end - self.previous_end, STEP_SECONDS, rel_tol=0, abs_tol=1e-6,
        ) or context != self.previous_context:
            self.runs.clear()
            self.active_pattern, self.pattern_streak = None, 0
        self.previous_end, self.previous_context = end, context

        measurements = {}
        for metric in JOYSTICK_METRICS:
            value = (values or {}).get(metric)
            baseline = self.baseline[metric]
            valid = value is not None and math.isfinite(value) and value >= 0
            difference, z, robust_z = deviations(value if valid else None, baseline)
            z = z if z is not None and math.isfinite(z) else None
            robust_z = robust_z if robust_z is not None and math.isfinite(robust_z) else None
            direction = "uncertain"
            reasons = list(quality)
            if not boundaries_set:
                reasons.append("Deviation boundaries are unconfigured.")
            if not valid or z is None:
                reasons.append("Feature or baseline variability unavailable.")
            if not reasons:
                if abs(z) <= rules.baseline_band_abs_z:
                    direction = "baseline_like"
                elif abs(z) >= rules.change_abs_z:
                    direction = "increased" if z > 0 else "decreased"
                else:
                    reasons.append("Deviation falls between baseline and change boundaries.")
            previous, count = self.runs.get(metric, (None, 0))
            count = (count + 1 if direction == previous else 1) if direction in ("increased", "decreased") else 0
            self.runs[metric] = direction, count
            state = direction
            if count and count < rules.consecutive_windows:
                state = "uncertain"
                reasons.append(f"Change persistence pending ({count}/{rules.consecutive_windows}).")
            measurements[metric] = {
                "value": value if valid else None, "baseline": asdict(baseline), "difference": difference,
                "z_score": z, "robust_z_score": robust_z, "observed_change": direction, "change_state": state,
                "consecutive_changes": count, "reasons": reasons,
            }

        reasons = list(quality)
        if not rules.higher_pattern or not rules.mapping_evidence.strip():
            reasons.append("No explicitly justified behavioural interpretation mapping.")
        if any(context.get(key) != value for key, value in rules.mapping_context.items()):
            reasons.append("Task context is unavailable or outside the mapping's documented context.")
        votes = {}
        for family, expected in rules.higher_pattern.items():
            metric = rules.family_metrics[family]
            observed = measurements[metric]["observed_change"]
            if observed == "uncertain":
                reasons.append(f"{family}: " + " ".join(measurements[metric]["reasons"]))
            elif observed == "baseline_like":
                votes[family] = "baseline_like"
            else:
                votes[family] = "higher_arousal_candidate" if observed == expected else "lower_arousal_candidate"

        counts = Counter(vote for vote in votes.values() if vote != "baseline_like")
        ranked = counts.most_common()
        if len(ranked) > 1 and (rules.conflict_policy != "majority_direction" or ranked[0][1] == ranked[1][1]):
            reasons.append("Mapped behavioural families conflict.")
        if ranked and "baseline_like" in votes.values() and rules.conflict_policy == "require_full_pattern":
            reasons.append("The full configured pattern is absent.")
        state = "uncertain"
        candidate = ranked[0][0] if not reasons and ranked else None
        # Hold the same family pattern for the full run, including which families are neutral.
        pattern = tuple(measurements[rules.family_metrics[family]]["observed_change"] for family in rules.higher_pattern)
        self.pattern_streak = (
            self.pattern_streak + 1 if pattern == self.active_pattern else 1
        ) if candidate else 0
        self.active_pattern = pattern if candidate else None
        if candidate:
            if self.pattern_streak >= rules.consecutive_windows:
                state = candidate
            else:
                reasons.append(f"Pattern persistence pending ({self.pattern_streak}/{rules.consecutive_windows}).")
        elif not reasons and votes and all(vote == "baseline_like" for vote in votes.values()):
            state = "baseline_like"
        return {
            "type": "experimental_joystick_window", "start_pc_s": start, "end_pc_s": end,
            "state": state, "task_context": context, "timing_status": timing_status,
            "coverage_percent": coverage, "sample_count": len(samples), "quality_flags": quality,
            "features": measurements, "family_votes": votes, "pattern_streak": self.pattern_streak,
            "rules": asdict(rules), "reasons": reasons,
        }
