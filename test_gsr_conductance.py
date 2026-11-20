"""Checks for optional, non-clinical Seeed Grove GSR conductance estimates."""

import math
import unittest

from gsr_conductance import (
    estimated_conductance_us,
    estimated_resistance_ohms,
    summarize_conductance,
    validate_calibration_adc,
)
from gsr_features import Sample, calculate_windows, estimate_windows


class ConductanceTests(unittest.TestCase):
    def test_seeed_equation_with_provisional_reference(self):
        self.assertAlmostEqual(estimated_resistance_ohms(495, 649), 130779.22077922078)
        self.assertAlmostEqual(estimated_conductance_us(495, 649), 7.646469215491559, places=4)

    def test_derived_conductance_is_positive_and_increases_as_adc_falls(self):
        self.assertGreater(estimated_conductance_us(495, 649), estimated_conductance_us(570, 649))
        self.assertGreater(estimated_conductance_us(570, 649), 0)

    def test_undefined_or_negative_resistance_is_not_reported(self):
        for raw in (649, 650, 1023):
            with self.subTest(raw=raw):
                self.assertIsNone(estimated_resistance_ohms(raw, 649))
                self.assertIsNone(estimated_conductance_us(raw, 649))

    def test_invalid_inputs_are_rejected(self):
        for calibration in (0, -1, 1024, True, 649.5):
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                validate_calibration_adc(calibration)
        for raw in (-1, 1024, True, 500.5):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                estimated_resistance_ohms(raw, 649)

    def test_summary_converts_samples_before_averaging(self):
        estimated, valid, invalid = summarize_conductance([495, 570], 649)
        self.assertEqual((valid, invalid), (2, 0))
        self.assertAlmostEqual(
            estimated,
            (estimated_conductance_us(495, 649) + estimated_conductance_us(570, 649)) / 2,
        )
        self.assertNotAlmostEqual(estimated, estimated_conductance_us(532, 649), places=3)

    def test_invalid_sample_invalidates_entire_estimated_window(self):
        mean, valid, invalid = summarize_conductance([495, 649, 570], 649)
        self.assertIsNone(mean)
        self.assertEqual((valid, invalid), (2, 1))
        self.assertEqual(summarize_conductance([], 649), (None, 0, 0))

    def test_window_boundaries_and_baseline_delta(self):
        samples = [Sample(100, 0, 495), Sample(100.25, 250, 500), Sample(100.5, 500, 570)]
        baseline = [Sample(90, 0, 560), Sample(90.25, 250, 560)]
        windows = calculate_windows(samples, window_seconds=.5, step_seconds=.25)
        estimates = estimate_windows(samples, windows, 649, baseline)
        self.assertEqual(len(estimates), 1)
        window = estimates[0]
        self.assertEqual((window.start_s, window.end_s, window.sample_count), (0, .5, 3))
        self.assertEqual((window.valid_conductance_samples, window.invalid_conductance_samples), (3, 0))
        self.assertEqual(window.calibration_adc, 649)
        self.assertAlmostEqual(
            window.mean_estimated_conductance_us,
            sum(estimated_conductance_us(x, 649) for x in (495, 500, 570)) / 3,
        )
        self.assertAlmostEqual(
            window.difference_from_baseline_estimated_conductance_us,
            window.mean_estimated_conductance_us - estimated_conductance_us(560, 649),
        )

    def test_invalid_baseline_disables_estimated_delta_not_raw_values(self):
        samples = [Sample(100, 0, 495), Sample(101, 1000, 500)]
        windows = calculate_windows(samples, window_seconds=1)
        baseline = [Sample(90, 0, 649)]
        estimate = estimate_windows(samples, windows, 649, baseline)[0]
        self.assertIsNotNone(estimate.mean_gsr_raw)
        self.assertIsNotNone(estimate.mean_estimated_conductance_us)
        self.assertIsNone(estimate.baseline_mean_estimated_conductance_us)
        self.assertIsNone(estimate.difference_from_baseline_estimated_conductance_us)


if __name__ == "__main__":
    unittest.main()
