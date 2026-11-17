import csv
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import features
import gsr_features
from experiment_session import (
    BASELINE_FILENAME, METRICS, Session, Stage, Summary, Workflow, analyze_session,
    compare_windows, create_baseline, deviations, latest_baseline, participant_directory,
)
from fuse_features import FusedWindow
from recording_clock import CLOCK_SOURCE, metadata_path


def window(raw, reversals=2, rate=None):
    return FusedWindow(100, 130, 600, 240, 30, 0.25, reversals, 20, rate, 0.5, 10, 100 / 3, raw)


class BaselineTests(unittest.TestCase):
    session = Session("P001", "baseline", Path("baseline_example"))

    def test_create_baseline_statistics_and_missing_feature(self):
        baseline = create_baseline(self.session, [window(400), window(500), window(700)])
        self.assertEqual(baseline.participant_id, "P001")
        self.assertEqual(baseline.window_count, 3)
        raw = baseline.metrics["mean_gsr_raw"]
        self.assertAlmostEqual(raw.mean, 1600 / 3)
        self.assertEqual((raw.median, raw.mad, raw.iqr), (500, 100, 150))
        self.assertAlmostEqual(raw.stddev, math.sqrt(140000 / 9))
        self.assertEqual(baseline.metrics["reversals_per_active_minute"], Summary(0, None, None, None, None, None))

    def test_insufficient_baseline_windows(self):
        with self.assertRaisesRegex(ValueError, "at least 3 valid fused windows"):
            create_baseline(self.session, [window(400), window(500)])

    def test_comparison_normalizes_only_with_defined_scale(self):
        baseline = create_baseline(self.session, [window(400), window(500), window(600)])
        row, = compare_windows([window(650, reversals=3, rate=5)], baseline)
        offsets = dict(zip(METRICS, (row[len(FusedWindow._fields) + 1 + i:][:3] for i in range(0, len(METRICS) * 3, 3))))
        difference, z, robust = offsets["mean_gsr_raw"]
        self.assertEqual(difference, 150)
        self.assertAlmostEqual(z, 150 / math.sqrt(20000 / 3))
        self.assertAlmostEqual(robust, 150 / 148.26)
        self.assertEqual(offsets["reversals"], (1, None, None))
        self.assertEqual(offsets["reversals_per_active_minute"], (None, None, None))
        self.assertEqual(deviations(500, Summary(2, 450, 450, 50, 50, 50)), (50, None, None))


class WorkflowTests(unittest.TestCase):
    def test_task_requires_baseline_and_invalid_transition_is_rejected(self):
        flow = Workflow()
        with self.assertRaisesRegex(ValueError, "valid participant baseline"):
            flow.start("task", False)
        with self.assertRaisesRegex(ValueError, "Cannot move"):
            flow.move(Stage.ANALYZING)
        self.assertIs(flow.stage, Stage.IDLE)

    def test_capture_analysis_retry_and_task(self):
        flow = Workflow()
        flow.start("baseline", False)
        with self.assertRaisesRegex(ValueError, "Cannot move"):
            flow.start("baseline", False)
        for stage in (
            Stage.RECORDING, Stage.STOPPING, Stage.AWAITING_XBOX, Stage.ANALYZING,
            Stage.AWAITING_XBOX, Stage.ANALYZING, Stage.IDLE,
        ):
            flow.move(stage)
        flow.start("task", True)
        flow.move(Stage.STOPPING)
        flow.move(Stage.IDLE)
        self.assertIs(flow.stage, Stage.IDLE)

    def test_participant_paths_reject_traversal_and_windows_devices(self):
        self.assertEqual(participant_directory("p001").name, "P001")
        for invalid in ("../P001", "P/001", "CON", "COM6", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                participant_directory(invalid)


class SessionIntegrationTests(unittest.TestCase):
    def write_recording(self, path, columns, rows):
        with path.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        metadata_path(path).write_text(json.dumps({
            "source": CLOCK_SOURCE, "host": "synthetic-host", "origin_unix_s": 1000,
        }), encoding="utf-8")

    def test_archive_baseline_and_analyze_task_with_unchanged_raw_csvs(self):
        parent = Path(tempfile.gettempdir()).resolve()
        root = (parent / f"teleop-test-{uuid4().hex}").resolve()
        self.assertEqual(root.parent, parent)
        root.mkdir()
        self.addCleanup(shutil.rmtree, root)
        with patch("experiment_session.PARTICIPANTS_DIR", root):
            baseline_session = Session.create("p001", "baseline")
            xbox = root / "xbox.csv"
            self.write_recording(xbox, features.REQUIRED_COLUMNS, [
                (100 + i / 16, 0.5, 0, 0, 0, 0, 0, 0, 0) for i in range(801)
            ])
            before = xbox.read_bytes()
            gsr = baseline_session.directory / "gsr_baseline.csv"
            self.write_recording(gsr, gsr_features.COLUMNS, [
                (102 + i / 8, i * 125, 400 + i) for i in range(361)
            ])
            gsr_before = gsr.read_bytes()
            profile_path = analyze_session(baseline_session, xbox, None, self.fail)
            self.assertEqual(profile_path.name, BASELINE_FILENAME)
            self.assertEqual(xbox.read_bytes(), before)
            self.assertEqual(gsr.read_bytes(), gsr_before)
            self.assertEqual((baseline_session.directory / "xbox" / xbox.name).read_bytes(), before)
            self.assertEqual(latest_baseline("P001")[0], profile_path)
            baseline = latest_baseline("P001")[1]
            self.assertEqual(baseline.window_count, 4)

            task_session = Session.create("P001", "task")
            task_gsr = task_session.directory / "gsr_task.csv"
            self.write_recording(task_gsr, gsr_features.COLUMNS, [
                (103 + i / 8, i * 125, 700) for i in range(241)
            ])
            output = analyze_session(task_session, xbox, baseline, self.fail)
            with output.open(newline="", encoding="utf-8") as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row["baseline_session"], baseline_session.directory.name)
            self.assertEqual(float(row["mean_gsr_raw_difference_from_mean"]), 120)
            self.assertEqual((task_session.directory / "baseline_used.json").read_bytes(), profile_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
