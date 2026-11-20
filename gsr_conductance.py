"""Optional Grove GSR v1.2 conversion using Seeed's documented resistance equation.

The calibration ADC is provisional until the manufacturer's minimum-output
procedure and measurement accuracy have been verified. These calculations are
not validated measurements of arousal or mental workload.
"""

import statistics
from collections.abc import Iterable

ADC_MAX = 1023
ADC_DENOMINATOR = 1024
RESISTOR_OHMS = 10000


def validate_calibration_adc(calibration_adc: int) -> None:
    """Require a positive 10-bit unworn reference."""
    if type(calibration_adc) is not int or not 1 <= calibration_adc <= ADC_MAX:
        raise ValueError("calibration ADC must be an integer between 1 and 1023.")


def estimated_resistance_ohms(raw_adc: int, calibration_adc: int) -> float | None:
    """Return Seeed's resistance estimate or None if raw >= calibration.

    For raw >= calibration the equation would be undefined or negative; do not
    interpret such results as physical resistance. Preserve the original ADC.
    """
    validate_calibration_adc(calibration_adc)
    if type(raw_adc) is not int or not 0 <= raw_adc <= ADC_MAX:
        raise ValueError("raw ADC must be an integer between 0 and 1023.")
    if raw_adc >= calibration_adc:
        return None
    return ((ADC_DENOMINATOR + 2 * raw_adc) * RESISTOR_OHMS) / (calibration_adc - raw_adc)


def estimated_conductance_us(raw_adc: int, calibration_adc: int) -> float | None:
    """Return an *estimated* conductance in microsiemens, not ground truth."""
    resistance = estimated_resistance_ohms(raw_adc, calibration_adc)
    return 1_000_000 / resistance if resistance is not None else None


def summarize_conductance(
    raw_values: Iterable[int], calibration_adc: int,
) -> tuple[float | None, int, int]:
    """Mean of per-sample estimates; invalidate the mean if any sample fails.

    A window with invalid samples must not silently drop them, which would
    bias the average. Counts are still reported to support quality review.
    """
    validate_calibration_adc(calibration_adc)
    estimates = []
    invalid = 0
    for raw_adc in raw_values:
        estimate = estimated_conductance_us(raw_adc, calibration_adc)
        if estimate is None:
            invalid += 1
        else:
            estimates.append(estimate)
    mean = statistics.mean(estimates) if estimates and invalid == 0 else None
    return mean, len(estimates), invalid
