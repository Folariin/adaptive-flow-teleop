import io
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import features
import gsr_features
from experiment_session import Baseline, Summary
from fuse_features import fuse_recordings
from live_features import CsvTail, RollingStream, format_comparison, monitor_task


class LiveTests(unittest.TestCase):
    def test_clock_mismatch_blocks_xbox_but_gsr_keeps_updating(self):
        root = Path("task")
        gsr = root / "gsr_example.csv"
        xbox = root / "xbox_live" / "controller_inputs_example.csv"
        stop = threading.Event()
        results = []

        def publish(update):
            results.append(update)
            stop.set()

        with (
            patch.object(Path, "glob", side_effect=[[gsr], [xbox]]),
            patch("live_features.verify_shared_clock", side_effect=ValueError("different machines")) as verify,
            patch.object(CsvTail, "update", return_value=("Coverage OK", None)),
        ):
            monitor_task(root, None, stop, publish)
        verify.assert_called_once_with(xbox, gsr)
        self.assertEqual(results[0]["gsr"], "Coverage OK")
        self.assertIn("different machines", results[0]["xbox"])

    def test_rolling_values_match_offline_at_identical_bounds(self):
        xbox = [features.Sample(100 + i / 16, ((i % 9 - 4) / 4, 0, 0, 0), i % 9 == 4) for i in range(561)]
        gsr = [gsr_features.Sample(100 + i / 8, 5000 + i * 125, 400 + i) for i in range(281)]
        offline = fuse_recordings(xbox, gsr)
        for is_gsr, rows, cutoff in ((False, xbox, 480), (True, gsr, 240)):
            stream = RollingStream(is_gsr)
            stream.append(rows[:cutoff])
            self.assertIsNone(stream.update(130)[1])
            stream.append(rows[cutoff:cutoff + 1])
            first = stream.update(130)[1]
            self.assertEqual(first, {key: getattr(offline[0], key) for key in first})
            self.assertEqual(stream.update(131)[1], first)
            stream.append(rows[cutoff + 1:])
            second = stream.update(135)[1]
            self.assertEqual(second, {key: getattr(offline[1], key) for key in second})

    def test_gap_invalidates_only_its_own_windows_and_stream_can_recover(self):
        stream = RollingStream(False)
        stream.append(features.Sample(100 + i / 16, (0, 0, 0, 0), True) for i in range(481) if i != 32)
        self.assertIn("Missing", stream.update(130)[0])
        stream.append(features.Sample(100 + i / 16, (0, 0, 0, 0), True) for i in range(481, 561))
        self.assertEqual(stream.update(135)[1]["idle_percent"], 100)
        self.assertIn("stale", stream.update(138)[0])

    def test_clock_discontinuities_and_future_timestamps(self):
        stream = RollingStream(True)
        stream.append([gsr_features.Sample(100, 0, 600)])
        self.assertIn("incompatible", stream.update(90)[0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            stream.append([gsr_features.Sample(100, 125, 600)])
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            stream.append([gsr_features.Sample(102, 125, 600)])

    def test_live_comparison_uses_saved_scales(self):
        baseline = Baseline("P001", "baseline_test", 3, {
            "mean_gsr_raw": Summary(3, 500, 500, 25, 20, 40),
        })
        output = format_comparison("Coverage OK", {"mean_gsr_raw": 550}, baseline)
        self.assertEqual(output.splitlines()[-1].split(), ["mean_gsr_raw", "550", "500", "50", "2", "1.686"])

    def test_csv_tail_waits_for_complete_rows_and_does_not_repeat_them(self):
        tail = CsvTail(Path("gsr.csv"), True)
        first = b"pc_perf_counter_s,arduino_millis,gsr_raw\r\n100,5000,600\r\n100.125,5125,"
        with (
            patch.object(Path, "open", side_effect=[io.BytesIO(first), io.BytesIO(first + b"601\r\n")]),
            patch("live_features.time.perf_counter", return_value=100.125),
        ):
            tail.update()
            self.assertEqual(tail.stream.timestamps, [100])
            tail.update()
            self.assertEqual(tail.stream.timestamps, [100, 100.125])


if __name__ == "__main__":
    unittest.main()
