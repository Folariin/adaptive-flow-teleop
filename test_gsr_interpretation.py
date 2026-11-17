import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiment_session import Baseline, Summary
from features import Sample as XboxSample
from gsr_diagnostics import Recording, inspect_signal
from gsr_features import Sample
from gsr_interpretation import Interpreter, Rules, State
from replay_session import ReplaySession


def samples(value, start=100):
    return [Sample(start + i / 8, i * 125, value) for i in range(241)]


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        self.summary = Summary(5, 500, 500, 10, 6, 12)
        self.quality, _ = inspect_signal(Recording(samples(500), 0, [], 0))
        self.rules = Rules("higher_counts", "Synthetic fixture only", 0.5, 2, 0.1, 98, 1)

    def evaluate(self, interpreter, value, start=100):
        return interpreter.evaluate(start, start + 30, value, samples(value, start) if value is not None else [], "test")

    def test_explicit_direction_and_baseline_band(self):
        for direction, value, state in (
            ("higher_counts", 525, State.HIGHER), ("higher_counts", 475, State.LOWER),
            ("lower_counts", 475, State.HIGHER), ("lower_counts", 525, State.LOWER),
            ("higher_counts", 504, State.BASELINE), ("higher_counts", 510, State.UNCERTAIN),
        ):
            with self.subTest(direction=direction, value=value):
                rules = replace(self.rules, direction=direction)
                result = self.evaluate(Interpreter(self.summary, self.quality, rules), value)
                self.assertEqual(result["state"], state.value)
                self.assertEqual(result["difference_adc"], value - 500)
                self.assertEqual(result["z_score"], (value - 500) / 10)

    def test_unknown_direction_and_unstable_or_missing_baseline(self):
        cases = (
            (self.quality, replace(self.rules, direction="unknown"), "direction"),
            (self.quality, replace(self.rules, direction_evidence=""), "calibration"),
            (replace(self.quality, drift_raw_per_s=0.2), self.rules, "drift exceeds"),
            (None, self.rules, "diagnostics are unavailable"),
        )
        for quality, rules, reason in cases:
            result = self.evaluate(Interpreter(self.summary, quality, rules), 525)
            self.assertEqual(result["state"], State.UNCERTAIN.value)
            self.assertIn(reason, " ".join(result["reasons"]))

    def test_invalid_quality_and_unavailable_normalization(self):
        interpreter = Interpreter(self.summary, self.quality, self.rules)
        for rows, value, flag in (
            ([], None, "missing_samples"), (samples(1023), 1023, "adc_rail"),
            (samples(1200), 1200, "invalid_adc_mean"),
            (samples(525)[:10], 525, "coverage"), (samples(525)[:20] + samples(525)[23:], 525, "suspect_intervals"),
        ):
            result = interpreter.evaluate(100, 130, value, rows, "test")
            self.assertEqual(result["state"], State.UNCERTAIN.value)
            self.assertIn(flag, " ".join(result["reasons"]))
        result = self.evaluate(Interpreter(replace(self.summary, stddev=0), self.quality, self.rules), 525)
        self.assertIsNone(result["z_score"])
        self.assertEqual(result["state"], State.UNCERTAIN.value)

    def test_sustained_deviation_resets_on_invalid_or_missing_windows(self):
        interpreter = Interpreter(self.summary, self.quality, replace(self.rules, consecutive_windows=2))
        states = [self.evaluate(interpreter, value, start)["state"] for start, value in (
            (100, 525), (105, None), (110, 525), (115, 525), (125, 525), (130, 475), (135, 475),
        )]
        self.assertEqual(states, [state.value for state in (
            State.UNCERTAIN, State.UNCERTAIN, State.UNCERTAIN, State.HIGHER,
            State.UNCERTAIN, State.UNCERTAIN, State.LOWER,
        )])

    def test_replay_audits_each_window_once_and_preserves_existing_panels(self):
        xbox = [XboxSample(100 + i / 16, (0, 0, 0, 0), True) for i in range(641)]
        gsr = [Sample(100 + i / 8, i * 125, 525) for i in range(321)]
        baseline = Baseline("P001", "synthetic", 5, {"mean_gsr_raw": self.summary})
        baseline.metrics.update({name: self.summary for name in (
            "movement_variability", "reversals", "reversals_per_active_minute",
            "input_roughness_per_s", "idle_seconds", "idle_percent",
        )})
        interpreter = Interpreter(self.summary, self.quality, replace(self.rules, consecutive_windows=2))
        replay = ReplaySession(xbox, gsr, baseline, interpreter, Path("audit.jsonl"))
        original = ReplaySession(xbox, gsr, baseline)
        writes = []

        class Audit(io.StringIO):
            def close(self):
                writes.append(self.getvalue())
                super().close()

        with patch.object(Path, "open", side_effect=lambda *args, **kwargs: Audit()):
            for session in (replay, original):
                session.resume(500)
            self.assertEqual(replay.tick(540), original.tick(540))
            replay.tick(550)
        rows = [json.loads(line) for line in writes]
        self.assertEqual([row["end_pc_s"] for row in rows], [130, 135, 140])
        self.assertEqual([row["state"] for row in rows], [State.UNCERTAIN, State.HIGHER, State.HIGHER])
        self.assertIn("Experimental interpretation", replay.interpretation_text())
