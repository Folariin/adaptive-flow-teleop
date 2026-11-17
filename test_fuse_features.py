import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

import features
import gsr_features
from fuse_features import coverage_slice, export_windows, fuse_recordings, main
from recording_clock import CLOCK_SOURCE, ClockMetadata, metadata_path, verify_shared_clock


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.xbox = [features.Sample(100 + i / 16, (0, 0, 0, 0), True) for i in range(801)]
        self.gsr = [gsr_features.Sample(103.125 + i / 8, i * 125, 300 + i) for i in range(361)]

    def test_different_starts_align_on_absolute_timestamps(self):
        windows = fuse_recordings(self.xbox, self.gsr)
        self.assertEqual([row.start_pc_s for row in windows], [103.125, 108.125, 113.125, 118.125])
        first, second = windows[:2]
        self.assertEqual((first.end_pc_s, first.xbox_sample_count, first.gsr_sample_count), (133.125, 481, 241))
        self.assertEqual((first.mean_gsr_raw, second.mean_gsr_raw), (420, 460))
        self.assertEqual((first.movement_variability, first.reversals, first.input_roughness_per_s), (0, 0, 0))
        self.assertEqual((first.idle_seconds, first.idle_percent), (30, 100))
        self.assertIsNone(first.reversals_per_active_minute)

    def test_missing_samples_only_exclude_affected_windows(self):
        xbox = [s for s in self.xbox if s.timestamp != 104.125]
        gsr = [s for s in self.gsr if s.pc_seconds != 111.125]
        with redirect_stderr(io.StringIO()) as stderr:
            windows = fuse_recordings(xbox, gsr)
        self.assertEqual([row.start_pc_s for row in windows], [113.125, 118.125])
        self.assertIn("1 Xbox and 1 GSR suspect sample intervals", stderr.getvalue())
        self.assertIn("skipped 2 of 4 windows", stderr.getvalue())

    def test_gaps_crossing_window_edges_count_toward_coverage(self):
        timestamps = [0, 1, 2, 3, 4]
        self.assertIsNone(coverage_slice(timestamps, [1], 0.5, 3))
        self.assertIsNone(coverage_slice(timestamps, [4], 1, 3.5))
        self.assertEqual(coverage_slice(timestamps, [1, 4], 1, 3), slice(1, 4))

    def test_nonoverlap_short_overlap_and_bad_arduino_timing(self):
        for shift, error in ((100, "do not overlap"), (30, "At least 30 s")):
            shifted = [sample._replace(pc_seconds=sample.pc_seconds + shift) for sample in self.gsr]
            with self.subTest(shift=shift), self.assertRaisesRegex(ValueError, error):
                fuse_recordings(self.xbox, shifted)
        bad_clock = [sample._replace(arduino_millis=sample.arduino_millis * 2) for sample in self.gsr]
        with self.assertRaisesRegex(ValueError, "GSR clock timing is unreliable"):
            fuse_recordings(self.xbox, bad_clock)

    def test_local_gsr_interval_disagreement_excludes_affected_window(self):
        gsr = self.gsr.copy()
        gsr[8] = gsr[8]._replace(pc_seconds=gsr[8].pc_seconds - 0.080)
        with redirect_stderr(io.StringIO()) as stderr:
            windows = fuse_recordings(self.xbox, gsr)
        self.assertEqual(windows[0].start_pc_s, 108.125)
        self.assertIn("2 GSR suspect sample intervals", stderr.getvalue())

    def test_cli_requires_clock_evidence_before_loading_recordings(self):
        with (
            patch("recording_clock.read_clock_metadata", return_value=None),
            patch("features.read_recording") as read_xbox,
            redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as exit_code,
        ):
            main(["xbox.csv", "gsr.csv"])
        self.assertEqual(exit_code.exception.code, 1)
        self.assertIn("Cannot verify a shared PC clock", stderr.getvalue())
        read_xbox.assert_not_called()

    def test_cli_exports_combined_rows(self):
        with (
            patch("recording_clock.read_clock_metadata", return_value=ClockMetadata("host", 1000)),
            patch("features.read_recording", return_value=self.xbox),
            patch("gsr_features.read_recording", return_value=self.gsr),
            patch("fuse_features.ANALYSIS_DIR", Path("analysis")),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(Path, "open", mock_open()) as open_csv,
            redirect_stdout(io.StringIO()) as stdout,
        ):
            self.assertEqual(main(["xbox.csv", "gsr.csv"]), 0)
        mkdir.assert_called_once_with(exist_ok=True)
        open_csv.assert_called_once_with("x", newline="", encoding="utf-8")
        rows = list(csv.DictReader(call.args[0] for call in open_csv().write.call_args_list))
        self.assertEqual((rows[0]["start_pc_s"], rows[0]["mean_gsr_raw"]), ("103.125", "420"))
        self.assertIn("Exported 4 synchronized windows", stdout.getvalue())


class RecordingClockTests(unittest.TestCase):
    xbox = Path("xbox.csv")
    gsr = Path("gsr.csv")

    def test_matching_sidecars_verify(self):
        self.assertEqual(metadata_path(self.xbox), Path("xbox.clock.json"))
        records = [
            io.StringIO(json.dumps({"source": CLOCK_SOURCE, "host": "same-host", "origin_unix_s": origin}))
            for origin in (1000, 1000.5)
        ]
        with patch.object(Path, "open", side_effect=records):
            verify_shared_clock(self.xbox, self.gsr)

    def test_host_or_clock_origin_mismatch_cannot_be_overridden(self):
        for clock, error in (
            (ClockMetadata("other-host", 1000), "different Windows machines"),
            (ClockMetadata("same-host", 1002), "clock origins disagree"),
        ):
            with (
                self.subTest(clock=clock),
                patch("recording_clock.read_clock_metadata", side_effect=[ClockMetadata("same-host", 1000), clock]),
                self.assertRaisesRegex(ValueError, error),
            ):
                verify_shared_clock(self.xbox, self.gsr, confirm_shared_clock=True)

    def test_legacy_confirmation_prints_warning(self):
        with patch.object(Path, "open", side_effect=FileNotFoundError), redirect_stderr(io.StringIO()) as stderr:
            verify_shared_clock(self.xbox, self.gsr, confirm_shared_clock=True)
        self.assertIn("confirmed by the operator", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
