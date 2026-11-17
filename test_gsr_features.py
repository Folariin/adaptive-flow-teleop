import csv
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import mock_open, patch

from gsr_features import Sample, Window, calculate_windows, export_windows, main, read_recording, warn_about_timing


HEADER = "pc_perf_counter_s,arduino_millis,gsr_raw\n"


class GsrFeatureTests(unittest.TestCase):
    def read_csv(self, contents):
        with patch.object(Path, "open", return_value=io.StringIO(contents)) as open_csv:
            samples = read_recording(Path("synthetic.csv"))
        open_csv.assert_called_once_with(newline="", encoding="utf-8-sig")
        return samples

    def test_default_windows_use_pc_time_and_include_both_boundaries(self):
        samples = [
            Sample(100 + offset, 1000 + index, raw)
            for index, (offset, raw) in enumerate(((0, 200), (5, 400), (7.5, 600), (30, 800), (35, 1000)))
        ]
        self.assertEqual(calculate_windows(samples, baseline_mean=650), [
            Window(0, 30, 4, 500, 650, -150),
            Window(5, 35, 4, 700, 650, 50),
        ])

    def test_short_recording_and_unsampled_complete_window(self):
        samples = [Sample(50, 0, 200), Sample(56, 6000, 400)]
        self.assertEqual(calculate_windows(samples), [])
        self.assertEqual(calculate_windows(samples, window_seconds=2, step_seconds=2), [
            Window(0, 2, 1, 200, None, None),
            Window(2, 4, 0, None, None, None),
            Window(4, 6, 1, 400, None, None),
        ])
        with self.assertRaisesRegex(ValueError, "Window step must be positive and finite"):
            calculate_windows(samples, step_seconds=float("nan"))

    def test_read_validation(self):
        cases = (
            ("", "empty"),
            (HEADER, "no samples"),
            ("pc_perf_counter_s,gsr_raw\n", "missing required columns: arduino_millis"),
            (HEADER + "nan,0,100\n", "invalid timestamp"),
            (HEADER + "1,0,100\n1,125,101\n", "strictly increasing"),
            (HEADER + "1,0,100\n0.9,125,101\n", "strictly increasing"),
            (HEADER + "1,0,1024\n", "between 0 and 1023"),
        )
        for contents, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.read_csv(contents)

    def test_timing_warnings_for_both_clocks_and_elapsed_disagreement(self):
        samples = self.read_csv(HEADER + "10,1000,600\n10.1,2500,601\n10.8,53,602\n")
        with redirect_stderr(io.StringIO()) as stderr:
            warn_about_timing(Path("synthetic.csv"), samples)
        warnings = stderr.getvalue()
        for message in (
            "Arduino timestamp gap of 1500 ms",
            "PC sampling gap of 0.700 s",
            "non-increasing Arduino timestamp (2500 -> 53 ms)",
            "elapsed times disagree",
            "synthetic.csv: timing issues may make analysis unreliable",
        ):
            self.assertIn(message, warnings)

    def test_cli_baseline_mean_and_custom_windows(self):
        recording = HEADER + "100,5000,400\n100.25,5250,600\n100.5,5500,800\n"
        baseline = HEADER + "200,1000,200\n200.25,1250,600\n"
        with (
            patch.object(Path, "open", side_effect=[io.StringIO(recording), io.StringIO(baseline)]),
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            status = main(["recording.csv", "--baseline", "resting.csv", "--window-seconds", "0.5", "--step-seconds", "0.25"])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertIn("Resting baseline: 400.0000 raw ADC counts (2 samples)", stdout.getvalue())
        self.assertIn("0.0000 | 0.5000 | 3 | 600.0000 | 200.0000", stdout.getvalue())

    def test_export_to_analysis_with_baseline_values(self):
        folder = Path("analysis")
        with (
            patch("gsr_features.ANALYSIS_DIR", folder),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(Path, "open", mock_open()) as open_csv,
        ):
            result = export_windows(Path("synthetic.csv"), [Window(0, 30, 3, 600, 400, 200)])
        mkdir.assert_called_once_with(exist_ok=True)
        open_csv.assert_called_once_with("x", newline="", encoding="utf-8")
        self.assertEqual(result.parent, folder)
        self.assertTrue(result.name.endswith("_gsr_windows.csv"))
        csv_lines = [call.args[0] for call in open_csv().write.call_args_list]
        row, = csv.DictReader(csv_lines)
        self.assertEqual(row["sample_count"], "3")
        self.assertEqual(row["mean_gsr_raw"], "600")
        self.assertEqual(row["baseline_mean_gsr_raw"], "400")
        self.assertEqual(row["difference_from_baseline_raw"], "200")


if __name__ == "__main__":
    unittest.main()
