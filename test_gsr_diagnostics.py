import io
import unittest
from pathlib import Path
from unittest.mock import patch

from gsr_diagnostics import Recording, inspect_signal, read_diagnostic_recording, window_features
from gsr_features import Sample, calculate_windows


class DiagnosticTests(unittest.TestCase):
    def test_linear_trend_and_existing_window_values(self):
        samples = [Sample(500 + i / 8, 1000 + 125 * i, 300 + i) for i in range(281)]
        recording = Recording(samples, 0, [], 0)
        quality, gaps = inspect_signal(recording)
        self.assertEqual((quality.duration_s, quality.median_rate_hz, quality.covered_percent), (35, 8, 100))
        self.assertAlmostEqual(quality.drift_raw_per_s, 8)
        windows = list(window_features(recording, quality, gaps, 400))
        expected = calculate_windows(samples, baseline_mean=400)
        self.assertEqual(len(windows), 2)
        for actual, original in zip(windows, expected):
            self.assertEqual({key: actual[key] for key in original._fields}, original._asdict())
            self.assertAlmostEqual(actual["trend_raw_per_s"], 8)
            self.assertAlmostEqual(actual["within_window_sd_raw"], (240 * 242 / 12) ** 0.5)
            self.assertEqual(actual["quality"], "no_flag")

    def test_invalid_rows_gaps_clock_disagreement_and_adc_rails(self):
        contents = (
            "pc_perf_counter_s,arduino_millis,gsr_raw\n"
            "100,1000,0\n100.125,1125,invalid\n100.25,1250,1023\n"
            "100.375,1000,600\n101,8000,600\nbad,9000,600\n"
        )
        with patch.object(Path, "open", return_value=io.StringIO(contents)) as opened:
            recording = read_diagnostic_recording(Path("source.csv"))
        opened.assert_called_once_with(encoding="utf-8-sig", newline="")
        quality, gaps = inspect_signal(recording)
        self.assertEqual((quality.valid_samples, quality.invalid_rows, quality.invalid_timestamps), (4, 2, 1))
        self.assertEqual((quality.rail_samples, quality.pc_gap_count, quality.arduino_gap_count), (2, 2, 2))
        self.assertEqual(quality.arduino_nonincreasing_count, 1)
        self.assertEqual(quality.max_elapsed_disagreement_s, 6)
        self.assertEqual(quality.covered_percent, 0)
        self.assertEqual(gaps, [(100, 100.25), (100.25, 100.375), (100.375, 101)])

    def test_gap_invalidates_only_windows_it_intersects(self):
        samples = [Sample(100 + i / 8, 125 * i, 600) for i in range(281) if i != 8]
        recording = Recording(samples, 0, [], 0)
        quality, gaps = inspect_signal(recording)
        first, second = window_features(recording, quality, gaps, 600)
        self.assertEqual(first["quality"], "gap_or_missing_data")
        self.assertEqual(second["quality"], "no_flag")
        self.assertEqual(second["difference_from_baseline_raw"], 0)
        self.assertEqual(second["within_window_sd_raw"], 0)

    def test_single_sample_and_short_recording(self):
        recording = Recording([Sample(100, 0, 600)], 0, [], 0)
        quality, gaps = inspect_signal(recording)
        self.assertIsNone(quality.drift_raw_per_s)
        self.assertIsNone(quality.covered_percent)
        self.assertFalse(list(window_features(recording, quality, gaps, 600)))
