# adaptive-flow-teleop

## Offline controller analysis

Run these commands from the repository root with a CSV recorded by
`xbox_controller.py`.

```powershell
python features.py "data/recording.csv"
python features.py "data/recording.csv" --sliding
python features.py "data/recording.csv" --sliding --window-seconds 10 --step-seconds 5 --export --plot
python -m unittest -v test_features
```

The default sliding window is 30 seconds with a 5-second step. Start and end
times are seconds since the first `perf_counter_s` sample. A window is complete
when its nominal end is at or before the last timestamp in the recording.
Samples exactly on either boundary are included. A recording shorter than the
window yields no complete windows.

Each window uses the whole-recording formulas on its selected samples.
Observed duration is the span from the first to last selected sample, which can
be shorter than the nominal window. An interval counts toward idle or active
time according to its earlier sample. Only intervals with both endpoints in
the window count, including the interval that ends at the last selected
sample. No time is added before the first or after the last selected sample.
Reversals and input changes also use only samples within the window.
Windows with no samples and metrics with a zero denominator display `N/A`.

`--export` saves a timestamped window CSV in `analysis/`, with blanks for
unavailable values. `--plot` works with either mode and saves a standalone SVG
in the same folder. Open the SVG in a browser to inspect the four deadzone-adjusted
joystick axes against elapsed time. The plot uses the full recording.

## Arduino GSR logging

Connect the Grove GSR sensor to A0. Open
[`gsr_sensor/gsr_sensor.ino`](gsr_sensor/gsr_sensor.ino) in Arduino IDE,
select **Arduino Uno** and **COM6**, and click **Upload**. If Arduino IDE asks
for a board package, install **Arduino AVR Boards**. Close the Serial
Monitor and Serial Plotter before running the logger from the repository root.

```powershell
python -m pip install pyserial
python gsr_logger.py --port COM6 --baud 9600
```

COM6 and 9600 baud are the defaults. The sketch starts idle. Python waits two
seconds for the Uno to settle after opening the port, clears the receive
buffer, and sends `START,<session-id>`. It retries while waiting for the
matching `ACK,<session-id>` and gives up after five seconds. The sketch then
sends `DATA,<session-id>,<arduino_millis>,<gsr_raw>` at roughly 8 Hz.
Python ignores samples before the matching acknowledgment and discards
samples from other sessions.

Before creating the CSV, Python waits for three consecutive sample intervals
between 75 and 200 ms on both clocks, with at most 50 ms disagreement per
interval. It reports the number of startup samples discarded and records
from the sample that completes the stability check. If sampling does not
stabilize within ten seconds, it ends the session without creating a CSV.

After startup, each valid sample becomes one row in `data/gsr_*.csv`.
`arduino_millis` is the original acquisition timestamp from the sketch.
`pc_perf_counter_s` is the PC's monotonic timestamp when the logger finishes
reading the line. Press Ctrl+C to flush the CSV and send
`STOP,<session-id>`. If the link drops, Python preserves the rows already
received and attempts STOP before closing the port. If no valid samples arrive
for five seconds, the logger ends the session.

The logger warns once per issue for duplicate or backward Arduino timestamps,
gaps over 500 ms between consecutive Arduino timestamps, or a difference over
1 second between the Arduino and PC elapsed times measured from the first
logged sample. It keeps valid rows and both timestamps as recorded. Review
timing warnings before synchronizing a recording with joystick data.

## Offline GSR analysis

```powershell
python gsr_features.py "data/gsr_recording.csv"
python gsr_features.py "data/gsr_recording.csv" --window-seconds 20 --step-seconds 5 --baseline "data/gsr_resting.csv" --export
python -m unittest -v test_gsr_features
```

By default, the analyzer calculates a raw ADC mean for each complete
30-second window, advancing 5 seconds at a time. Windows use the PC
timestamps relative to the first recorded sample and include both endpoints.
Each selected sample has equal weight. A complete window with no samples
has a count of zero and no mean. A recording shorter than the requested
window yields no results.

The optional resting baseline is the arithmetic mean of all samples in the
baseline CSV. The reported difference is the window mean minus the baseline
mean. `--export` writes a separate timestamped window CSV under `analysis/`.
The input files are opened read-only. Values are raw ADC counts, with no
calibration to skin conductance or validation as workload scores.

The analyzer rejects non-increasing PC timestamps. It checks both recordings
for PC or Arduino gaps greater than 0.5 seconds, non-increasing Arduino
timestamps, and disagreement greater than 1 second between the clocks'
elapsed times from the first sample. These issues produce warnings and leave
the source rows intact.

## Fuse concurrent Xbox and GSR recordings

Use the same Windows computer and boot for both loggers, with CPython 3.10 or
later. Keep the Arduino Serial Monitor closed. In one terminal, start GSR and
wait for its logging message. In a second terminal, start the Xbox logger as
usual. Record at least 30 seconds of overlap. Stop Xbox with B, then stop GSR
with Ctrl+C. Keep each new CSV with its accompanying `.clock.json` file.

```powershell
# Terminal 1
python gsr_logger.py --port COM6 --baud 9600
# Terminal 2
python xbox_controller.py
# After capture
python fuse_features.py "data/controller_inputs_....csv" "data/gsr_....csv"
```

The sidecars provide a hashed Windows machine identifier, the PC clock source,
and a wall-clock estimate of the PC counter origin. Fusion requires matching
machines and origins within one second. It uses the original absolute
`perf_counter` timestamps and applies no offset. For legacy files without
sidecars, pass `--confirm-shared-clock` only if you verified that both captures
ran on the same Windows boot with CPython 3.10 or later. Numeric overlap alone
cannot establish clock compatibility.

The first 30-second window starts at the later of the two first timestamps.
Windows advance by five seconds and include both boundaries. A window is
excluded if a sample interval crossing it exceeds 90 ms on Xbox or 225 ms on
either GSR clock. Non-increasing Arduino timestamps and Arduino/PC interval
differences over 75 ms also invalidate the affected windows. An Arduino/PC
elapsed-time difference over one second from the start of GSR capture rejects
the pair. Gaps indicate suspected missing or delayed samples.

The exported `analysis/fused_features_*.csv` includes absolute window bounds,
sample counts, the existing Xbox features, and mean raw GSR. Xbox features
use their observed sample span and GSR means weight each in-window sample
equally. Source files are opened read-only.

## Windows experiment controller

Launch from the repository root on the same Windows PC used for capture.

```powershell
python -m pip install pyserial
python experiment_controller.py
```

Enter a pseudonymous participant ID, such as `P001`. Session files go under
`data/participants/P001/`. Use the same participant ID for the baseline and
task. The app starts and stops GSR on COM6. It displays readiness after the
Arduino handshake and stable GSR sampling, and stores GSR diagnostics in
`gsr_process.log` in each session folder.

1. Select **Start Baseline**. Once GSR is recording, the operator manually
   starts `python xbox_controller.py` in a separate terminal and runs the
   standardized, familiar teleoperation task. Xbox capture is coupled to
   robot control, so the app leaves both startup and shutdown to the operator.
2. Capture at least 40 seconds of clean overlap to get three complete
   30-second windows with a 5-second step. Press **B** to close the Xbox
   capture, then select **Stop & Save Baseline** in the app and choose the
   closed Xbox CSV. Keep its clock sidecar beside it.
3. After the baseline is saved, select **Start Task** and repeat the manual
   Xbox startup for the actual task. Capture at least 30 seconds of clean
   overlap. Press **B**, select **Stop & Analyze Task**, and choose the Xbox
   CSV. **View Results** opens the latest result file.

The app archives the selected Xbox CSV and clock sidecar without altering the
originals. Each session includes the GSR CSV and its sidecar. The app uses
the existing shared-clock and coverage checks to produce `fused_features.csv`.
Baseline sessions save `baseline_profile.json`. Task sessions save
`task_comparison.csv` and a snapshot of the baseline used.

The baseline describes valid overlapping windows from the familiar task. It
stores each feature's count, arithmetic mean, median, population standard
deviation, median absolute deviation (MAD), and interquartile range. Task
results include the difference from the baseline mean, the difference divided
by the population standard deviation, and a robust deviation from the median
divided by `1.4826 × MAD`. Normalized fields are blank when the baseline has
fewer than three valid values for that feature or the relevant scale is zero.
The overlapping windows are descriptive observations. GSR values remain raw
ADC counts, and the comparisons provide no workload or arousal classification.

During a main task, wait for **GSR is recording**, click **Copy Task Xbox
Command**, and paste the command into the terminal where you normally run
LeRobot. The command adds `--live-output-dir` to share the Xbox CSV in this
task's folder. The baseline workflow still uses the usual Xbox command.
Keep the live output on the local disk. The app reads complete rows from the
two CSVs in a background thread. Capture flushes periodically, so the display
can lag the latest sample briefly.

Each live panel uses its own first sample on the absolute PC clock to anchor
30-second windows at five-second steps. The Xbox panel checks both clock
sidecars before showing comparisons. The panels show their window bounds
and sample coverage separately. A gap invalidates only the windows that cross
it, and a stream that stops publishing clears its comparison after two
seconds. Offline fusion still uses common overlapping windows, so its
window boundaries can differ from the live panels. The calculations agree
when the samples and bounds are the same.

For a hardware check, use a saved baseline, start a task, and record for at
least 40 seconds after manually starting Xbox. Confirm both panels show
comparisons and then advance by five seconds. Press **B** in the Xbox
terminal, then select **Stop & Analyze Task** and choose the CSV from
`xbox_live/`. The live tables contain feature values and descriptive
baseline deviations. They do not estimate physiological arousal.

### Replay a saved session

Run `python experiment_controller.py` from the repository root. From the idle
screen, click **Replay Session**. Select a saved baseline JSON and the GSR and
Xbox CSVs from the same concurrent recording. Clock sidecars are loaded next
to the CSVs by default. Use the two optional clock fields when the corresponding
`.clock.json` files are stored elsewhere. Replay requires verified matching
clock sidecars and overlapping recordings. Playback reads the files without
opening any device or starting either logger.

Click **Start**. Replay runs at the original pace, starting at the earliest PC
timestamp and preserving the delay before the second recording begins.
**Pause** freezes the virtual clock, **Resume** continues from that position,
and **Stop** resets playback so you can start again. After the first complete
30-second window, the two panels show feature values and baseline deviations
at five-second steps. Watch the progress bar and original PC window bounds to
verify playback.

Replay uses the later of the two first timestamps as the window origin, like
offline fusion. At identical bounds, valid per-stream features and deviations
use the same samples and formulas as offline analysis, with rounding in the
display. A panel can still report a valid window when the other stream has a
gap or has already ended. Offline fusion exports only windows valid in both.
Replay retains each panel's last evaluated window at end of file. Live capture
anchors each panel to its own first sample, so its window bounds can differ.

Run the hardware-free playback tests with
`python -B -m unittest -v test_replay_session test_live_features`.

### GSR diagnostics for P001

Use the standardized teleoperation baseline CSV as the reference. Run from
the repository root in PowerShell.

```powershell
$baseline = "data/participants/P001/baseline_20261106_202110_344575/gsr_20261106_202113_456559.csv"
$task = "data/participants/P001/task_20261106_202743_478838/gsr_20261106_202746_617227.csv"
python gsr_diagnostics.py $task --baseline $baseline
python -m pip install matplotlib
python gsr_diagnostics.py $task --baseline $baseline --plot
python -B -m unittest -v test_gsr_diagnostics test_gsr_features
```

The command prints diagnostics for both recordings and writes a separate CSV
in `analysis/`. With `--plot` it also saves one PNG per recording, showing raw
samples, existing 30-second means at their window midpoints, the baseline
reference, shaded suspect intervals, and a recording-wide linear fit.

- The reference is the arithmetic mean across the valid raw samples in the
  baseline GSR CSV, consistent with the GSR-only baseline option in
  `gsr_features.py`. The experiment profile uses the mean of fused window
  means and can give a different reference. The diagnostic delta is the
  window mean minus the raw recording reference, in ADC counts.
- Trend is an ordinary least-squares slope on actual PC elapsed time, in
  counts per second. Variability is the population standard deviation of
  the observed counts. The whole-recording slope describes possible drift.
  A slope can also reflect changes during the task.
- Coverage is the percentage of the usable first-to-last PC time span bridged
  by nonsuspect intervals. Suspect intervals include gaps over 225 ms on either
  clock, non-increasing Arduino timestamps, interval disagreement over 75 ms,
  and identifiable invalid ADC rows between valid samples. Median cadence
  is the reciprocal of the median PC intersample interval. Coverage is a
  timing diagnostic and does not verify electrode contact.
- Malformed data rows are counted and excluded from feature calculations.
  Invalid timestamps that cannot be placed on the time axis and cumulative
  clock disagreement over 1 second flag the recording's windows. Counts at
  either ADC rail (0 or 1023) remain visible and flag possible clipping.
  The logger can discard invalid input before CSV output, so CSV diagnostics
  count only invalid rows that appear in the file. Treat flagged windows as
  descriptive and inspect the samples and baseline quality before using them.

The [Seeed Grove sensor documentation](https://wiki.seeedstudio.com/Grove-GSR_Sensor/)
describes an analog voltage output from resistance sensing. The nominal 8 Hz
recording can show changes over several seconds, but these files lack a
documented calibration and a validated way to distinguish phasic responses
from grip changes, electrode motion, and electronics artifacts. Response
counts and amplitudes are therefore withheld. Established electrodermal
measures include skin conductance level and skin conductance response
amplitude in microsiemens, as described in the [SPR publication
recommendations](https://doi.org/10.1111/j.1469-8986.2012.01384.x).
The diagnostic mean, signed delta, slope, and variability describe raw ADC
readings. The sign indicates the direction of count change and carries no
established physiological interpretation for this setup.

### Experimental GSR interpretation in replay

Run `python experiment_controller.py`, open **Replay Session**, and choose
the participant baseline profile and concurrent GSR and Xbox CSVs as usual.
For P001, use `baseline_used.json` and the GSR CSV in
`data/participants/P001/task_20261106_202743_478838/`, plus the CSV in its
`xbox/` directory. Replay locates the raw GSR baseline in the profile's
baseline session. Keep that session directory with the profile when moving
the recordings. Missing baseline diagnostics produce uncertainty.

Leave **GSR interpretation rules** blank to inspect the conservative default.
The GSR interpretation panel displays the current mean in ADC counts, the
saved profile mean, the signed difference, the normalized deviation, and
the evidence behind its state. Each replay writes a separate
`analysis/*_interpretation.jsonl` audit with window bounds, measurements,
baseline and signal diagnostics, rule settings, and reasons. The existing
GSR and joystick replay comparisons remain available in their own panels.

An optional JSON file can supply these experimental settings. The following
is a disabled configuration, equivalent to leaving the field blank.

```json
{
  "direction": "unknown",
  "direction_evidence": "",
  "baseline_band_abs_z": null,
  "candidate_abs_z": null,
  "max_baseline_abs_drift_raw_per_s": null,
  "min_coverage_percent": null,
  "consecutive_windows": 1
}
```

Use `higher_counts` or `lower_counts` only after establishing which count
direction corresponds to increased conductance for the actual setup.
`direction_evidence` should identify the calibration record. The software
requires that reference but cannot verify the calibration itself.
Set both z boundaries, a maximum absolute baseline drift, and a minimum
coverage percentage explicitly. The candidate boundary must exceed the
baseline band. The rule uses the ordinary z-score against the saved fused
GSR baseline summary. Robust z is included in the audit for inspection.

Windows inside the band are `baseline_like`. Windows beyond the candidate
boundary can become `higher_arousal_candidate` or `lower_arousal_candidate`
after the configured number of successive valid windows in the same
direction. Missing windows, quality failures, and changes of direction
reset that sequence. The intermediate band is `uncertain`. Coverage uses
the observed usable span divided by the nominal 30 seconds, so inclusive
sample selection can leave a small fraction uncovered at the boundaries.
Successive windows overlap and are statistically dependent.

Before using candidates to interpret participant studies, establish the
response direction and usable range with bench calibration. Quantify drift,
repeatability, and motion or grip artifacts under the actual task setup.
Compare against calibrated electrodermal measurements and independently
defined experimental conditions. Specify the rule before evaluation and
assess errors and the uncertainty rate on held-out sessions. Calibration
establishes the electrical response, while the experimental comparison is
needed to assess whether the candidate labels support an arousal claim.

### Joystick change detection in replay

Run `python experiment_controller.py` and open **Replay Session**. For the
saved P001 task, select `baseline_used.json` and the GSR CSV from
`data/participants/P001/task_20261106_202743_478838/`, then select the Xbox
CSV under that folder's `xbox/` directory. Leave the clock overrides blank
if the sidecars are beside the CSVs. Press **Start** and wait for the first
complete 30-second window. Windows advance every five seconds.

The **Joystick behaviour** panel shows the existing features and baseline
comparisons. The **Joystick experimental interpretation** panel shows the
quality checks, feature-level changes, and candidate state. The joystick
audit in `analysis/*_joystick_interpretation.jsonl` records each window
separately from the GSR audit. Leave **Joystick rules and context** blank to
inspect the default, which reports `uncertain` with no configured rule.

To exercise behavioural changes only, save the following as
`analysis/joystick_change_demo.json` and select it in **Joystick rules and
context**. These values are software demonstration settings with no
physiological validation.

```json
{
  "baseline_band_abs_z": 0.5,
  "change_abs_z": 2.0,
  "consecutive_windows": 2,
  "min_coverage_percent": 98,
  "higher_pattern": {},
  "mapping_evidence": "",
  "task_context": {}
}
```

Feature changes use the saved baseline's ordinary z-score. The audit also
includes the signed raw difference and robust z-score when available.
An unchanged feature within the band reports `baseline_like`. A change
beyond the configured boundary reports `increased` or `decreased` once
the direction persists for the configured number of windows. The space
between the boundaries, a zero baseline standard deviation, a missing
measurement, or a failed coverage check yields `uncertain` for that
feature. Coverage measures the observed sample span relative to 30 seconds.
The replay coverage checks also reject timestamp gaps.

To evaluate an experimental arousal mapping, supply `higher_pattern` as a
mapping from a behavioural family to `increased` or `decreased`, and cite
its justification in `mapping_evidence`. An opposing pattern corresponds to
the lower candidate. Supported families are `movement_variability`,
`reversals`, `input_roughness`, and `idle`. Each contributes at most one
direction. `family_metrics` can select reversal count or reversal rate
(default), and idle seconds or idle percentage (default). When overriding
it, list one existing metric for each of the four families. All six
metrics remain visible in the audit.

The default `conflict_policy`, `require_full_pattern`, requires all mapped
families to agree. `allow_baseline_like` permits neutral families alongside
an agreeing direction. `majority_direction` permits a directional majority
and abstains on ties. An unavailable mapped feature blocks the pattern in
all modes. The same observed pattern must persist across consecutive valid
windows. The windows overlap and do not provide independent confirmations.

Supply known `task_type` or `difficulty` as text in `task_context`.
`mapping_context` can require exact values for either field. A mismatch or
missing required context yields `uncertain`. The audit leaves unknown
context empty. Context changes reset persistence. Candidate states are
experimental interpretations of joystick behaviour. Changes in manipulation
demand and task strategy can also alter these features.

Run the targeted tests with
`python -B -m unittest -v test_joystick_interpretation test_replay_session test_gsr_interpretation test_live_features test_fuse_features`.
