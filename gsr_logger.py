"""Record raw Arduino GSR samples to a timestamped CSV."""

import argparse
import csv
import os
import sys
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from recording_clock import write_clock_metadata


DATA_DIR = Path(__file__).resolve().parent / "data"
COLUMNS = ("pc_perf_counter_s", "arduino_millis", "gsr_raw")
MAX_LINE_BYTES = 64
STARTUP_SETTLE_SECONDS = 2.0
START_TIMEOUT_SECONDS = 5.0
START_RETRY_SECONDS = 1.0
STABILIZATION_TIMEOUT_SECONDS = 10.0
STABLE_INTERVALS_REQUIRED = 3
MIN_SAMPLE_INTERVAL_SECONDS = 0.075
MAX_SAMPLE_INTERVAL_SECONDS = 0.200
MAX_INTERVAL_DISAGREEMENT_SECONDS = 0.050
SAMPLE_TIMEOUT_SECONDS = 5.0
MAX_ARDUINO_GAP_MS = 500
MAX_ELAPSED_DISAGREEMENT_SECONDS = 1.0


def parse_line(line: bytes) -> tuple[int, int] | None:
    """Return the Arduino timestamp and Uno ADC count, or skip a malformed line."""
    if len(line) > MAX_LINE_BYTES:
        return None
    millis_text, _, raw_text = line.strip().partition(b",")
    if not (millis_text.isdigit() and raw_text.isdigit()):
        return None

    millis, raw = int(millis_text), int(raw_text)
    if millis > 0xFFFFFFFF or raw > 1023:
        return None
    return millis, raw


class TimingChecks:
    """Warn once per issue while preserving the acquisition and reception times."""

    def __init__(self, pc_seconds: float, arduino_millis: int):
        self.first_pc_seconds = pc_seconds
        self.first_arduino_millis = arduino_millis
        self.previous_arduino_millis = arduino_millis
        self.warned = set()

    def warn(self, kind: str, message: str) -> None:
        if kind not in self.warned:
            print(f"Warning: {message}", file=sys.stderr)
            self.warned.add(kind)

    def observe(self, pc_seconds: float, arduino_millis: int) -> None:
        previous = self.previous_arduino_millis
        gap_ms = arduino_millis - previous
        if gap_ms <= 0:
            self.warn("order", f"non-increasing Arduino timestamp ({previous} -> {arduino_millis} ms).")
        elif gap_ms > MAX_ARDUINO_GAP_MS:
            self.warn("gap", f"Arduino timestamp gap of {gap_ms} ms ({previous} -> {arduino_millis} ms).")
        self.previous_arduino_millis = arduino_millis

        arduino_elapsed = (arduino_millis - self.first_arduino_millis) / 1000
        pc_elapsed = pc_seconds - self.first_pc_seconds
        if abs(arduino_elapsed - pc_elapsed) > MAX_ELAPSED_DISAGREEMENT_SECONDS:
            self.warn(
                "elapsed",
                f"elapsed times disagree since the first sample "
                f"(Arduino {arduino_elapsed:.3f} s, PC {pc_elapsed:.3f} s). "
                "Check for buffered serial data or irregular Arduino timestamps before synchronization.",
            )


def serial_lines(connection, stop_requested: threading.Event | None = None):
    """Yield timestamped complete lines, or None so callers can check deadlines."""
    pending = b""
    while True:
        if stop_requested is not None and stop_requested.is_set():
            raise KeyboardInterrupt
        chunk = connection.read_until(size=MAX_LINE_BYTES)
        if not chunk:
            yield None
            continue

        complete = chunk.endswith(b"\n")
        if complete:
            received_at = time.perf_counter()

        # A timeout can split a line. Once it is oversized, discard through LF.
        if pending is not None:
            pending += chunk
            if len(pending) > MAX_LINE_BYTES:
                pending = None
        if complete:
            message = (received_at, pending.rstrip(b"\r\n")) if pending is not None else None
            pending = b""
            yield message
        else:
            yield None


def start_session(connection, lines, session: bytes) -> None:
    command = b"START," + session + b"\n"
    expected_ack = b"ACK," + session
    next_retry = time.monotonic()
    deadline = next_retry + START_TIMEOUT_SECONDS
    while (now := time.monotonic()) < deadline:
        if now >= next_retry:
            connection.write(command)
            next_retry = now + START_RETRY_SECONDS
        message = next(lines)
        if message is not None and message[1] == expected_ack:
            return
    raise TimeoutError("No fresh START acknowledgment. Check that gsr_sensor.ino is uploaded and the baud rate matches.")


def session_samples(lines, session: bytes):
    """Yield valid samples for this session, or None for other reads."""
    prefix = b"DATA," + session + b","
    warned_invalid = False
    for message in lines:
        if message is None:
            yield None
            continue
        received_at, line = message
        if not line.startswith(prefix):
            yield None
            continue
        sample = parse_line(line[len(prefix):])
        if sample is None:
            if not warned_invalid:
                print("Warning: invalid GSR sample in the active session.", file=sys.stderr)
                warned_invalid = True
            yield None
            continue
        yield (received_at, *sample)


def wait_for_stable_sampling(samples):
    """Return the first recordable sample and the number withheld at startup."""
    deadline = time.monotonic() + STABILIZATION_TIMEOUT_SECONDS
    previous = None
    stable_intervals = 0
    discarded = 0
    while time.monotonic() < deadline:
        sample = next(samples)
        if sample is None:
            continue
        if previous is not None:
            pc_interval = sample[0] - previous[0]
            arduino_interval = (sample[1] - previous[1]) / 1000
            stable = (
                MIN_SAMPLE_INTERVAL_SECONDS <= pc_interval <= MAX_SAMPLE_INTERVAL_SECONDS
                and MIN_SAMPLE_INTERVAL_SECONDS <= arduino_interval <= MAX_SAMPLE_INTERVAL_SECONDS
                and abs(pc_interval - arduino_interval) <= MAX_INTERVAL_DISAGREEMENT_SECONDS
            )
            stable_intervals = stable_intervals + 1 if stable else 0
            if stable_intervals == STABLE_INTERVALS_REQUIRED:
                return sample, discarded
        previous = sample
        discarded += 1
    raise TimeoutError(f"GSR sampling did not stabilize. Discarded {discarded} startup samples.")


def log_samples(samples, output, first_sample: tuple[float, int, int]) -> None:
    writer = csv.writer(output)
    writer.writerow(COLUMNS)
    writer.writerow(first_sample)
    timing = TimingChecks(first_sample[0], first_sample[1])
    last_flush = first_sample[0]
    deadline = time.monotonic() + SAMPLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sample = next(samples)
        if sample is None:
            continue
        deadline = time.monotonic() + SAMPLE_TIMEOUT_SECONDS
        writer.writerow(sample)
        if sample[0] - last_flush >= 0.5:
            output.flush()
            last_flush = sample[0]
        timing.observe(sample[0], sample[1])
    raise TimeoutError(f"No valid GSR samples from the active session for {SAMPLE_TIMEOUT_SECONDS:g} seconds.")


def watch_stdin(stop_requested: threading.Event) -> None:
    """The desktop controller closes its pipe to request normal shutdown."""
    try:
        os.read(sys.stdin.fileno(), 1)
    except OSError:
        pass
    finally:
        stop_requested.set()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6", help="Arduino serial port (default: COM6)")
    parser.add_argument("--baud", type=int, default=9600, help="serial baud rate (default: 9600)")
    parser.add_argument("--output-dir", type=Path, help="directory for the CSV and clock sidecar (default: data/)")
    parser.add_argument("--stop-on-stdin", action="store_true", help="stop when the parent closes stdin")
    args = parser.parse_args(argv)
    if args.baud <= 0:
        parser.error("--baud must be positive")

    try:
        import serial
    except ImportError:
        print("Install pySerial with: python -m pip install pyserial", file=sys.stderr)
        return 1

    stop_requested = None
    if args.stop_on_stdin:
        stop_requested = threading.Event()
        threading.Thread(target=watch_stdin, args=(stop_requested,), daemon=True).start()
    output_dir = args.output_dir if args.output_dir is not None else DATA_DIR

    try:
        with serial.Serial(port=args.port, baudrate=args.baud, timeout=0.25, write_timeout=1.0) as connection:
            print(f"Waiting for the Arduino on {args.port} at {args.baud} baud.")
            if stop_requested is None:
                time.sleep(STARTUP_SETTLE_SECONDS)
            elif stop_requested.wait(STARTUP_SETTLE_SECONDS):
                raise KeyboardInterrupt
            connection.reset_input_buffer()
            session = uuid4().hex.encode("ascii")
            lines = serial_lines(connection, stop_requested)
            try:
                start_session(connection, lines, session)
                print("Waiting for stable GSR sample arrival.")
                samples = session_samples(lines, session)
                first_sample, discarded = wait_for_stable_sampling(samples)
                print(f"Discarded {discarded} startup samples before recording.")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = output_dir / f"gsr_{timestamp}.csv"
                with path.open("x", newline="", encoding="utf-8") as output:
                    print(f"Logging GSR to {path}. Press Ctrl+C to stop.")
                    write_clock_metadata(path)
                    log_samples(samples, output, first_sample)
            finally:
                try:
                    connection.write(b"STOP," + session + b"\n")
                except OSError as error:
                    print(f"Warning: could not send STOP on {args.port}: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nGSR logging stopped.")
        return 0
    except serial.SerialException as error:
        print(f"Serial error on {args.port}: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Invalid serial configuration: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"GSR logging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
