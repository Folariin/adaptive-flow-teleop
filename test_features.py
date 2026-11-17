import csv
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch
from xml.etree import ElementTree

from features import (
    REQUIRED_COLUMNS, Sample, calculate_features, calculate_windows, count_reversals,
    export_windows, main, plot_axes, read_recording,
)


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(REQUIRED_COLUMNS)
    writer.writerows(rows)
    return stream.getvalue()


class FeatureTests(unittest.TestCase):
    def read_csv(self, contents):
        with patch.object(Path, "open", return_value=io.StringIO(contents)) as open_file:
            samples = read_recording(Path("synthetic.csv"))
        open_file.assert_called_once_with(newline="", encoding="utf-8-sig")
        return samples

    def test_whole_recording_with_irregular_intervals(self):
        samples = self.read_csv(csv_text([
            (10.0, 1, -1, 0, 0, 0, 0, 0, 0),
            (10.25, 0, 0, 0, 0, 0, 0, 0, 0),
            (10.75, -1, 1, 0, 0, 0, 0, 0, 0),
            (12.0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]))
        result = calculate_features(samples)
        self.assertEqual(result.duration, 2)
        self.assertAlmostEqual(result.movement_variability, (0.5 ** 0.5) / 2)
        self.assertEqual(result.reversals, 2)
        self.assertEqual(result.active_seconds, 1.5)
        self.assertEqual(result.reversals_per_active_minute, 80)
        self.assertEqual(result.input_roughness, 0.75)
        self.assertEqual(result.idle_seconds, 0.5)
        self.assertEqual(result.idle_percent, 25)

    def test_reversal_threshold_and_neutral_timeout(self):
        timestamps = (0, 0.25, 0.75, 1, 1.75, 3, 3.25)
        axis = (0.05, -0.049, -0.05, 0, 0.05, -0.05, -0.6)
        # The first change is at the timeout boundary. The next neutral gap
        # is too long. The last change has no intervening neutral sample.
        self.assertEqual(count_reversals(timestamps, axis), 2)

    def test_triggers_buttons_and_small_processed_movement_are_active(self):
        rows = [
            (0, 0.01, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0.1, 0, 0, 0),
            (2, 0, 0, 0, 0, 0, 0.1, 0, 0),
            (3, 0, 0, 0, 0, 0, 0, 1, 0),
            (4, 0, 0, 0, 0, 0, 0, 0, 1),
            (5, 0, 0, 0, 0, 0, 0, 0, 0),
        ]
        result = calculate_features(self.read_csv(csv_text(rows)))
        self.assertEqual(result.active_seconds, 5)
        self.assertEqual(result.idle_seconds, 0)
        self.assertEqual(result.reversals_per_active_minute, 0)

    def test_all_idle_and_single_sample(self):
        rows = [(1, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0)]
        samples = self.read_csv(csv_text(rows))
        result = calculate_features(samples)
        self.assertEqual(result.idle_percent, 100)
        self.assertEqual(result.input_roughness, 0)
        self.assertIsNone(result.reversals_per_active_minute)

        result = calculate_features(samples[:1])
        self.assertEqual(result.movement_variability, 0)
        self.assertIsNone(result.input_roughness)
        self.assertIsNone(result.idle_percent)

    def test_missing_columns_empty_and_invalid_timestamps(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns.*rb_pressed"):
            self.read_csv("perf_counter_s\n1\n")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.read_csv("")
        with self.assertRaisesRegex(ValueError, "no samples"):
            calculate_features(self.read_csv(csv_text([])))

        inactive = (0,) * 8
        for timestamp in ("invalid", "nan", "inf"):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "Line 2.*perf_counter_s"):
                self.read_csv(csv_text([(timestamp, *inactive)]))
        for timestamp in (1, 0):
            samples = self.read_csv(csv_text([(1, *inactive), (timestamp, *inactive)]))
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "strictly increasing"):
                calculate_features(samples)

    def test_cli_summary_and_error(self):
        rows = [(1, 0, 0, 0, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0, 0, 0, 0)]
        with patch("sys.argv", ["features.py", "synthetic.csv"]):
            with patch.object(Path, "open", return_value=io.StringIO(csv_text(rows))):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(), 0)
            self.assertIn("Idle time: 1.0000 s (100.0000%)", output.getvalue())

            with patch.object(Path, "open", return_value=io.StringIO("")):
                with redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as exit_code:
                    main()
            self.assertEqual(exit_code.exception.code, 1)
            self.assertIn("The recording is empty.", error.getvalue())


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            Sample(100 + offset, (lx, 0, 0, 0), lx == 0)
            for offset, lx in ((0, 1), (1, 0), (1.5, -1), (3, 1), (4.5, 0), (6, -1))
        ]

    def test_overlapping_windows_include_both_boundaries(self):
        windows = calculate_windows(self.samples, window_seconds=3, step_seconds=1.5)
        self.assertEqual([(w.start, w.end) for w in windows], [(0, 3), (1.5, 4.5), (3, 6)])
        for window, selected in zip(windows, (self.samples[:4], self.samples[2:5], self.samples[3:])):
            self.assertEqual(window.features, calculate_features(selected))
        first = windows[0].features
        self.assertAlmostEqual(first.movement_variability, 11 ** 0.5 / 16)
        self.assertEqual(first.reversals, 2)
        self.assertAlmostEqual(first.input_roughness, 1 / 3)
        self.assertAlmostEqual(first.idle_percent, 100 * 0.5 / 3)

    def test_irregular_window_does_not_extend_to_either_boundary(self):
        first, second = calculate_windows(self.samples, window_seconds=3, step_seconds=2)
        self.assertEqual((first.start, first.end, second.start, second.end), (0, 3, 2, 5))
        result = second.features
        self.assertEqual(result.duration, 1.5)
        self.assertEqual(result.idle_percent, 0)
        self.assertEqual(result.reversals, 0)
        self.assertAlmostEqual(result.input_roughness, 1 / 6)

    def test_default_window_and_short_recording(self):
        self.assertEqual(calculate_windows(self.samples), [])
        samples = [Sample(t, (0, 0, 0, 0), True) for t in (100, 130, 135)]
        windows = calculate_windows(samples)
        self.assertEqual([(w.start, w.end) for w in windows], [(0, 30), (5, 35)])

    def test_complete_window_with_no_samples_and_invalid_parameters(self):
        samples = [self.samples[0], self.samples[-1]]
        windows = calculate_windows(samples, window_seconds=2, step_seconds=2)
        self.assertIsNone(windows[1].features)
        self.assertIsNone(windows[2].features.idle_percent)
        with self.assertRaisesRegex(ValueError, "Window step must be positive and finite"):
            calculate_windows(samples, step_seconds=0)

    def test_export_and_plot(self):
        recording = Path("synthetic.csv")
        windows = calculate_windows(self.samples, window_seconds=3, step_seconds=2)
        with patch("features.output_path", return_value=Path("windows.csv")):
            with patch.object(Path, "open", mock_open()) as open_file:
                export_windows(recording, windows)
        csv_rows = [call.args[0] for call in open_file().write.call_args_list]
        first, second = list(csv.DictReader(csv_rows))
        self.assertEqual(float(first["reversals"]), 2)
        self.assertEqual(float(second["observed_duration_s"]), 1.5)

        with patch("features.output_path", return_value=Path("joysticks.svg")):
            with patch.object(Path, "open", mock_open()) as open_file:
                plot_axes(recording, self.samples)
        svg = ElementTree.fromstring(open_file().write.call_args.args[0])
        lines = svg.findall(".//{http://www.w3.org/2000/svg}polyline")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].attrib["points"].split()[:3], ["80.00,55.00", "225.00,110.00", "297.50,165.00"])


if __name__ == "__main__":
    unittest.main()
