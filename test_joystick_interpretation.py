import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from experiment_session import Baseline, METRICS, Summary
from features import Sample
from fuse_features import fuse_recordings
from gsr_features import Sample as GsrSample
from joystick_interpretation import Interpreter, Rules
from live_features import JOYSTICK_METRICS
from replay_session import ReplaySession


class JoystickInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {metric: Summary(5, 10, 10, 2, 1, 3) for metric in METRICS}
        self.rules = Rules(
            baseline_band_abs_z=0.5, change_abs_z=2, min_coverage_percent=98,
            higher_pattern={"reversals": "increased", "idle": "decreased"},
            mapping_evidence="Synthetic test fixture", mapping_context={"task_type": "fixture"},
        )
        self.context = {"task_type": "fixture"}
        self.higher = dict.fromkeys(JOYSTICK_METRICS, 10)
        self.higher.update(reversals=5, reversals_per_active_minute=15, idle_seconds=15, idle_percent=5)

    def evaluate(self, interpreter, values, start=100, context=None, samples=None):
        if samples is None:
            samples = [Sample(start + i / 16, (0, 0, 0, 0), True) for i in range(481)]
        return interpreter.evaluate(start, start + 30, values, samples, "test", context or self.context)

    def test_related_metrics_are_audited_but_do_not_cast_extra_votes(self):
        higher = self.evaluate(Interpreter(self.baseline, self.rules), self.higher)
        self.assertEqual(higher["state"], "higher_arousal_candidate")
        self.assertEqual(higher["features"]["reversals"]["change_state"], "decreased")
        self.assertEqual(higher["features"]["reversals_per_active_minute"]["change_state"], "increased")
        self.assertEqual(higher["features"]["idle_percent"]["z_score"], -2.5)
        lower = self.evaluate(Interpreter(self.baseline, self.rules), {key: 20 - val for key, val in self.higher.items()})
        self.assertEqual(lower["state"], "lower_arousal_candidate")
        neutral = self.evaluate(Interpreter(self.baseline, self.rules), dict.fromkeys(JOYSTICK_METRICS, 10))
        self.assertEqual(neutral["state"], "baseline_like")
        self.assertTrue(all(item["change_state"] == "baseline_like" for item in neutral["features"].values()))

    def test_no_mapping_keeps_behaviour_and_arousal_separate(self):
        rules = replace(self.rules, higher_pattern={}, mapping_evidence="")
        record = self.evaluate(Interpreter(self.baseline, rules), self.higher)
        self.assertEqual(record["state"], "uncertain")
        self.assertEqual(record["features"]["idle_percent"]["change_state"], "decreased")
        self.assertIn("mapping", " ".join(record["reasons"]))

    def test_conflict_and_optional_neutral_family(self):
        rules = replace(self.rules, conflict_policy="allow_baseline_like")
        conflict = self.evaluate(Interpreter(self.baseline, rules), dict(self.higher, idle_percent=15))
        self.assertEqual(conflict["state"], "uncertain")
        self.assertIn("conflict", " ".join(conflict["reasons"]))
        mixed = dict(self.higher, idle_percent=10)
        self.assertEqual(self.evaluate(Interpreter(self.baseline, rules), mixed)["state"], "higher_arousal_candidate")
        self.assertEqual(self.evaluate(Interpreter(self.baseline, self.rules), mixed)["state"], "uncertain")
        majority = replace(rules, conflict_policy="majority_direction")
        self.assertEqual(self.evaluate(Interpreter(self.baseline, majority), dict(self.higher, idle_percent=15))["state"], "uncertain")
        majority = replace(majority, higher_pattern=dict(majority.higher_pattern, movement_variability="increased"))
        values = dict(self.higher, idle_percent=15, movement_variability=15)
        self.assertEqual(self.evaluate(Interpreter(self.baseline, majority), values)["state"], "higher_arousal_candidate")

    def test_bad_samples_and_zero_variability(self):
        self.baseline["idle_percent"] = replace(self.baseline["idle_percent"], stddev=0)
        interpreter = Interpreter(self.baseline, self.rules)
        record = self.evaluate(interpreter, self.higher)
        self.assertIsNone(record["features"]["idle_percent"]["z_score"])
        self.assertEqual(record["state"], "uncertain")
        for samples in ([], [Sample(100, (float("nan"), 0, 0, 0), True), Sample(130, (0, 0, 0, 0), True)]):
            self.assertTrue(self.evaluate(interpreter, self.higher, samples=samples)["quality_flags"])
        self.assertTrue(self.evaluate(interpreter, None)["quality_flags"])

    def test_persistence_resets_on_context_change_quality_and_skipped_window(self):
        rules = replace(self.rules, consecutive_windows=2)
        interpreter = Interpreter(self.baseline, rules)
        contexts_and_values = (
            (100, self.context, self.higher), (105, self.context, self.higher),
            (110, dict(self.context, difficulty="changed"), self.higher),
            (115, {"task_type": "other"}, self.higher), (120, self.context, self.higher),
            (125, self.context, None), (130, self.context, self.higher),
            (140, self.context, self.higher), (145, self.context, self.higher),
        )
        records = [self.evaluate(interpreter, values, start, context) for start, context, values in contexts_and_values]
        self.assertEqual([r["end_pc_s"] for r in records if r["state"] == "higher_arousal_candidate"], [135, 175])
        self.assertIn("context", " ".join(records[3]["reasons"]))
        self.assertEqual(records[0]["features"]["idle_percent"]["change_state"], "uncertain")
        self.assertEqual(records[1]["features"]["idle_percent"]["change_state"], "decreased")

    def test_replay_audit_matches_offline_and_leaves_existing_panels_intact(self):
        xbox = [Sample(100 + i / 16, (0, 0, 0, 0), True) for i in range(641)]
        gsr = [GsrSample(100 + i / 8, i * 125, 525) for i in range(321)]
        offline = fuse_recordings(xbox, gsr)
        profile = Baseline("P001", "synthetic", 5, self.baseline)
        replay = ReplaySession(
            xbox, gsr, profile, joystick_interpreter=Interpreter(self.baseline, Rules()),
            joystick_audit_path=Path("joystick.jsonl"),
        )
        original = ReplaySession(xbox, gsr, profile)
        writes = []

        class Audit(io.StringIO):
            def close(self):
                writes.append(self.getvalue())
                super().close()

        with patch.object(Path, "open", side_effect=lambda *args, **kwargs: Audit()):
            replay.resume(0)
            original.resume(0)
            self.assertEqual(replay.tick(40), original.tick(40))
            replay.tick(45)
        rows = [json.loads(line) for line in writes]
        self.assertEqual([row["end_pc_s"] for row in rows], [130, 135, 140])
        for row, expected in zip(rows, offline):
            self.assertEqual(row["state"], "uncertain")
            self.assertEqual({key: val["value"] for key, val in row["features"].items()},
                             {key: getattr(expected, key) for key in JOYSTICK_METRICS})
        self.assertIn("Experimental interpretation", replay.joystick_interpretation_text())
