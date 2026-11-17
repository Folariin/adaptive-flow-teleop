"""Play saved samples on a shared virtual PC clock without opening devices."""

import csv
import json
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path

import features
import gsr_features
from experiment_session import load_baseline
from gsr_diagnostics import inspect_signal, read_diagnostic_recording
from gsr_interpretation import Interpreter, Rules, load_rules
from joystick_interpretation import Interpreter as JoystickInterpreter
from joystick_interpretation import Rules as JoystickRules
from joystick_interpretation import load_configuration
from live_features import RollingStream, format_comparison
from recording_clock import verify_shared_clock


class ReplayStream:
    def __init__(self, rows, is_gsr, origin):
        self.rows = rows
        self.timestamps = [row.pc_seconds if is_gsr else row.timestamp for row in rows]
        self.rolling = RollingStream(is_gsr, window_origin=origin)
        self.index = 0
        self.error = None

    def advance(self, pc_time, on_window=None):
        if self.error:
            return self.error, None
        limit = bisect_right(self.timestamps, pc_time, lo=self.index)
        try:
            if on_window is not None and self.index == 0 and limit:
                self.rolling.append(self.rows[:1])
                self.index = 1
            # Evaluate each reachable boundary even if a UI tick delivers multiple windows.
            while on_window is not None and self.index < limit:
                boundary = bisect_left(self.timestamps, self.rolling.next_end, self.index, limit)
                if boundary == limit:
                    break
                self.rolling.append(self.rows[self.index:boundary + 1])
                self.index = boundary + 1
                while self.rolling.next_end <= self.timestamps[boundary]:
                    result = self.rolling.update(self.timestamps[boundary], latest_window=False)
                    on_window(result)
            self.rolling.append(self.rows[self.index:limit])
            self.index = limit
            ended = limit == len(self.rows)
            status, values = self.rolling.update(self.timestamps[-1] if ended else pc_time)
        except ValueError as error:
            self.error = f"Invalid recording: {error}"
            return self.error, None
        if ended:
            if self.rolling.latest is None:
                return "End of recording · Insufficient samples for a complete 30-second window.", None
            return f"End of recording · {status}", values
        return status, values


class ReplaySession:
    def __init__(
        self, xbox, gsr, baseline, interpreter=None, audit_path=None, *,
        joystick_interpreter=None, joystick_audit_path=None, task_context=None,
    ):
        xbox_times = features.recording_timestamps(xbox)
        self.first_pc = min(xbox_times[0], gsr[0].pc_seconds)
        self.last_pc = max(xbox_times[-1], gsr[-1].pc_seconds)
        overlap_start = max(xbox_times[0], gsr[0].pc_seconds)
        overlap_end = min(xbox_times[-1], gsr[-1].pc_seconds)
        if overlap_end <= overlap_start:
            raise ValueError("The recordings do not overlap on the shared PC clock.")
        self.gsr = ReplayStream(gsr, True, overlap_start)
        self.xbox = ReplayStream(xbox, False, overlap_start)
        self.baseline = baseline
        self.duration = self.last_pc - self.first_pc
        self.position = 0.0
        self.started_at = None
        self.interpreter = interpreter
        self.audit_path = audit_path
        self.interpretation = None
        self.interpretation_error = None
        self.gsr_status = ""
        self.joystick_interpreter = joystick_interpreter
        self.joystick_audit_path = joystick_audit_path
        self.task_context = task_context or {}
        self.joystick_interpretation = None
        self.joystick_error = None
        self.xbox_status = ""

    def record_joystick(self, result):
        start, end, samples = self.xbox.rolling.last_window
        status, values = result
        record = self.joystick_interpreter.evaluate(start, end, values, samples, status, self.task_context)
        if self.joystick_audit_path is not None:
            try:
                with self.joystick_audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            except OSError as error:
                self.joystick_error = f"Could not write joystick audit: {error}"
        self.joystick_interpretation = record

    def joystick_interpretation_text(self):
        if self.xbox.error or self.joystick_error:
            return f"uncertain · {self.xbox.error or self.joystick_error}"
        if self.xbox_status.startswith("Samples are stale"):
            return f"uncertain · {self.xbox_status}"
        record = self.joystick_interpretation
        if record is None:
            return self.xbox_status or "Waiting for the first joystick window."
        context = ", ".join(f"{key}={value}" for key, value in record["task_context"].items())
        coverage = record["coverage_percent"]
        lines = [
            f"Experimental interpretation · {record['state']}", record["timing_status"],
            f"Coverage {coverage:.2f}%" if coverage is not None else "Coverage unavailable",
            f"Task context · {context or 'unavailable'}",
        ]
        for metric, item in record["features"].items():
            observed = item["observed_change"]
            detail = f" (observed {observed})" if observed != item["change_state"] else ""
            lines.append(f"{metric}: {item['change_state']}{detail}")
        feature_reasons = dict.fromkeys(
            reason for item in record["features"].values() for reason in item["reasons"]
        )
        return "\n".join(lines + list(dict.fromkeys([*record["reasons"], *feature_reasons])))

    def record_interpretation(self, result):
        self.write_interpretation(*self.gsr.rolling.last_window, result)

    def write_interpretation(self, start, end, samples, result):
        status, values = result
        current = values["mean_gsr_raw"] if values is not None else None
        record = self.interpreter.evaluate(start, end, current, samples, status)
        if self.audit_path is not None:
            try:
                with self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            except OSError as error:
                self.interpretation_error = f"Could not write interpretation log: {error}"
        self.interpretation = record

    def interpretation_text(self):
        if self.gsr.error or self.interpretation_error:
            return f"uncertain · {self.gsr.error or self.interpretation_error}"
        record = self.interpretation
        if self.gsr_status.startswith("Samples are stale"):
            return f"uncertain · {self.gsr_status}"
        if record is None:
            return "Waiting for the first GSR window."
        def number(value):
            return "unavailable" if value is None else f"{value:.4g}"
        return "\n".join([
            f"Experimental interpretation · {record['state']}",
            record["timing_status"],
            f"Mean {number(record['raw_mean_adc'])} ADC · baseline {number(record['baseline_mean_adc'])} ADC · "
            f"change {number(record['difference_adc'])} ADC · z {number(record['z_score'])}",
            f"Coverage {number(record['coverage_percent'])}%",
            *record["reasons"],
        ])

    @property
    def finished(self):
        return self.position >= self.duration

    def resume(self, now):
        if self.started_at is None and not self.finished:
            self.started_at = now - self.position

    def pause(self, now):
        self.tick(now)
        self.started_at = None

    def tick(self, now):
        if self.started_at is not None:
            self.position = min(self.duration, max(self.position, now - self.started_at))
            if self.finished:
                self.started_at = None
        pc_time = self.last_pc if self.finished else self.first_pc + self.position
        gsr = self.gsr.advance(pc_time, self.record_interpretation if self.interpreter else None)
        self.gsr_status = gsr[0]
        if self.interpreter and self.gsr.error:
            rolling = self.gsr.rolling
            while rolling.next_end is not None and rolling.next_end <= min(pc_time, self.gsr.timestamps[-1]):
                end = rolling.next_end
                self.write_interpretation(end - gsr_features.DEFAULT_WINDOW_SECONDS, end, [], gsr)
                rolling.next_end += gsr_features.DEFAULT_STEP_SECONDS
        xbox = self.xbox.advance(pc_time, self.record_joystick if self.joystick_interpreter else None)
        self.xbox_status = xbox[0]
        return format_comparison(*gsr, self.baseline), format_comparison(*xbox, self.baseline)


def load_replay(
    baseline: Path, gsr: Path, xbox: Path,
    gsr_sidecar: Path | None = None, xbox_sidecar: Path | None = None,
    rules_path: Path | None = None, joystick_rules_path: Path | None = None, *, audit=False,
) -> ReplaySession:
    verify_shared_clock(xbox, gsr, xbox_sidecar=xbox_sidecar, gsr_sidecar=gsr_sidecar)
    xbox_samples = features.read_recording(xbox)
    gsr_samples = gsr_features.read_recording(gsr)
    profile = load_baseline(baseline)
    rules = load_rules(rules_path) if rules_path is not None else Rules()
    joystick_rules, context = load_configuration(joystick_rules_path) if joystick_rules_path else (JoystickRules(), {})
    baseline_folder = baseline.parent.parent / profile.session
    if baseline.parent.name == profile.session:
        baseline_folder = baseline.parent
    recordings = (
        list(baseline_folder.glob("gsr_*.csv"))
        if profile.session.startswith("baseline_") and Path(profile.session).name == profile.session else []
    )
    quality = None
    if len(recordings) == 1:
        try:
            quality, _ = inspect_signal(read_diagnostic_recording(recordings[0]))
        except (OSError, ValueError, csv.Error):
            pass
    session = ReplaySession(
        xbox_samples, gsr_samples, profile, Interpreter(profile.metrics["mean_gsr_raw"], quality, rules),
        joystick_interpreter=JoystickInterpreter(profile.metrics, joystick_rules), task_context=context,
    )
    if audit:
        folder = gsr_features.ANALYSIS_DIR
        folder.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session.audit_path = folder / f"{gsr.stem}_{stamp}_interpretation.jsonl"
        with session.audit_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "type": "replay_metadata", "participant": profile.participant_id,
                "baseline_profile": str(baseline), "baseline_gsr": [str(path) for path in recordings],
                "gsr": str(gsr), "rules_file": str(rules_path) if rules_path else None,
            }) + "\n")
        session.joystick_audit_path = folder / f"{xbox.stem}_{stamp}_joystick_interpretation.jsonl"
        with session.joystick_audit_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "type": "replay_metadata", "participant": profile.participant_id,
                "baseline_profile": str(baseline), "xbox": str(xbox),
                "rules_file": str(joystick_rules_path) if joystick_rules_path else None,
            }) + "\n")
    return session
