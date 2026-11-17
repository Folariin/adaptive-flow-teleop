import unittest
from pathlib import Path
from unittest.mock import patch

import features
import gsr_features
from experiment_session import Baseline, METRICS, summarize
from fuse_features import fuse_recordings
from replay_session import ReplaySession, load_replay


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.xbox = [features.Sample(100 + i / 16, ((i % 5) / 4, 0, 0, 0), i % 5 == 0) for i in range(641)]
        self.gsr = [gsr_features.Sample(102 + i / 8, 1000 + i * 125, 500 + i % 16) for i in range(281)]
        self.baseline = Baseline("P001", "synthetic", 3, {key: summarize([0, 1, 2]) for key in METRICS})

    def test_original_timestamps_and_pause_resume(self):
        replay = ReplaySession(self.xbox, self.gsr, self.baseline)
        replay.resume(500)
        replay.tick(501)
        self.assertEqual((replay.xbox.index, replay.gsr.index), (17, 0))
        replay.pause(501.5)
        replay.tick(800)
        self.assertEqual((replay.position, replay.xbox.index, replay.gsr.index), (1.5, 25, 0))
        replay.resume(900)
        replay.tick(900.5)
        self.assertEqual((replay.position, replay.xbox.index, replay.gsr.index), (2, 33, 1))

    def test_identical_window_bounds_match_offline_fusion(self):
        replay = ReplaySession(self.xbox, self.gsr, self.baseline)
        offline = fuse_recordings(self.xbox, self.gsr)
        replay.resume(500)
        for expected in offline:
            replay.tick(500 + expected.end_pc_s - replay.first_pc)
            for stream in (replay.xbox, replay.gsr):
                status, values = stream.rolling.latest
                self.assertIn(f"PC {expected.start_pc_s:.3f} to {expected.end_pc_s:.3f}", status)
                self.assertEqual(values, {key: getattr(expected, key) for key in values})

    def test_missing_samples_invalidate_only_affected_panel(self):
        xbox = [sample for sample in self.xbox if sample.timestamp != 110]
        replay = ReplaySession(xbox, self.gsr, self.baseline)
        replay.resume(500)
        gsr, joystick = replay.tick(532)
        self.assertIn("Coverage OK", gsr)
        self.assertIn("Missing or discontinuous", joystick)

    def test_eof_preserves_last_window_and_stops_clock(self):
        replay = ReplaySession(self.xbox, self.gsr, self.baseline)
        replay.resume(500)
        gsr, joystick = replay.tick(538)
        self.assertIn("End of recording", gsr)
        self.assertIn("Coverage OK", gsr)
        self.assertNotIn("End of recording", joystick)
        final = replay.tick(540)
        self.assertTrue(replay.finished)
        self.assertTrue(all("End of recording" in panel for panel in final))
        self.assertEqual(final, replay.tick(600))

    def test_short_recording_reports_insufficient_data_at_eof(self):
        replay = ReplaySession(self.xbox[:65], self.gsr[:9], self.baseline)
        replay.resume(500)
        self.assertTrue(all("Insufficient samples" in panel for panel in replay.tick(504)))

    def test_rejects_nonoverlap_and_requires_clock_verification(self):
        shifted = [row._replace(pc_seconds=row.pc_seconds + 100) for row in self.gsr]
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            ReplaySession(self.xbox, shifted, self.baseline)
        with (
            patch("replay_session.verify_shared_clock", side_effect=ValueError("clock mismatch")) as verify,
            patch("features.read_recording") as read,
            self.assertRaisesRegex(ValueError, "clock mismatch"),
        ):
            load_replay(Path("profile"), Path("gsr"), Path("xbox"), Path("gclock"), Path("xclock"))
        verify.assert_called_once_with(Path("xbox"), Path("gsr"), xbox_sidecar=Path("xclock"), gsr_sidecar=Path("gclock"))
        read.assert_not_called()
