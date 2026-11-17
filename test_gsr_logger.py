import csv
import io
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from gsr_logger import (
    MAX_LINE_BYTES, STARTUP_SETTLE_SECONDS, TimingChecks, log_samples, main, parse_line,
    serial_lines, watch_stdin,
)


SESSION_HEX = "a" * 32
SESSION = SESSION_HEX.encode("ascii")
STALE = b"b" * 32
ACK = b"ACK," + SESSION + b"\r\n"
START = b"START," + SESSION + b"\n"
STOP = b"STOP," + SESSION + b"\n"


def data(payload, session=SESSION):
    return b"DATA," + session + b"," + payload


class SerialError(OSError):
    pass


class CaptureBytes(io.BytesIO):
    def close(self):
        if not self.closed:
            self.contents = self.getvalue()
        super().close()


class GsrLoggerTests(unittest.TestCase):
    def test_parent_pipe_eof_requests_cooperative_stop(self):
        stop_requested = threading.Event()
        connection = MagicMock()
        with (
            patch("gsr_logger.sys.stdin", SimpleNamespace(fileno=lambda: 0)),
            patch("gsr_logger.os.read", return_value=b"") as read,
        ):
            watch_stdin(stop_requested)
        read.assert_called_once_with(0, 1)
        self.assertTrue(stop_requested.is_set())
        with self.assertRaises(KeyboardInterrupt):
            next(serial_lines(connection, stop_requested))
        connection.read_until.assert_not_called()

    def test_parse_valid_lines_and_boundaries(self):
        self.assertEqual(parse_line(b"54787,648\r\n"), (54787, 648))
        self.assertEqual(parse_line(b"4294967295,1023\n"), (4294967295, 1023))
        self.assertEqual(parse_line(b"0,0\n"), (0, 0))

    def test_parse_malformed_lines(self):
        for line in (
            b"", b"startup text\n", b"1,\n", b"1,2,3\n", b"1,\xff\n",
            b"-1,2\n", b"1,-2\n", b"1,1024\n", b"4294967296,2\n",
            b"1,2.5\n", b"0" * MAX_LINE_BYTES + b",1\n",
        ):
            with self.subTest(line=line):
                self.assertIsNone(parse_line(line))

    def run_mock_logger(
        self, args, chunks, pc_times=None, startup_lines=(ACK,), stop_error=None,
        stable_at=(10.0, 54787), expected_discarded=3,
    ):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        warmup_lines = []
        warmup_times = []
        if stable_at is not None:
            pc, millis = stable_at
            warmup_lines = [data(f"{millis - 125 * i},640\n".encode()) for i in (3, 2, 1)]
            warmup_times = [pc - 0.125 * i for i in (3, 2, 1)]
        connection.read_until.side_effect = [*startup_lines, *warmup_lines, *chunks]
        if stop_error is not None:
            connection.write.side_effect = [len(START), stop_error]
        serial = SimpleNamespace(Serial=MagicMock(return_value=connection), SerialException=SerialError)
        raw = CaptureBytes()
        output = io.TextIOWrapper(raw, encoding="utf-8", newline="")
        stderr = io.StringIO()
        data_dir = Path("data")
        if pc_times is None:
            pc_times = [10.0 + i / 8 for i in range(len(chunks))]
        with (
            patch.dict("sys.modules", {"serial": serial}),
            patch("gsr_logger.DATA_DIR", data_dir),
            patch(
                "gsr_logger.time.perf_counter",
                side_effect=[0.0] * len(startup_lines) + warmup_times + list(pc_times),
            ),
            patch("gsr_logger.uuid4", return_value=SimpleNamespace(hex=SESSION_HEX)),
            patch("gsr_logger.time.sleep") as sleep,
            patch("gsr_logger.write_clock_metadata") as write_metadata,
            patch.object(Path, "mkdir") as mkdir,
            patch.object(Path, "open", return_value=output, autospec=True) as open_csv,
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(stderr),
        ):
            startup = MagicMock()
            startup.attach_mock(serial.Serial, "open_port")
            startup.attach_mock(sleep, "sleep")
            startup.attach_mock(connection.reset_input_buffer, "reset_input_buffer")
            startup.attach_mock(connection.write, "write")
            startup.attach_mock(connection.read_until, "read_until")
            status = main(args)

        self.assertEqual(
            startup.mock_calls[1:5],
            [call.sleep(STARTUP_SETTLE_SECONDS), call.reset_input_buffer(), call.write(START),
             call.read_until(size=MAX_LINE_BYTES)],
        )
        self.assertEqual(connection.write.call_args_list, [call(START), call(STOP)])
        self.assertIn(f"Discarded {expected_discarded} startup samples", stdout.getvalue())
        connection.reset_input_buffer.assert_called_once_with()
        mkdir.assert_called_once_with(exist_ok=True)
        path, mode = open_csv.call_args.args
        self.assertEqual(mode, "x")
        self.assertEqual(path.parent, data_dir)
        self.assertRegex(path.name, r"^gsr_\d{8}_\d{6}_\d{6}\.csv$")
        write_metadata.assert_called_once_with(path)
        self.assertTrue(output.closed)
        self.assertTrue(raw.closed)
        connection.__exit__.assert_called_once()
        rows = list(csv.reader(io.StringIO(raw.contents.decode("utf-8"))))
        self.assertEqual(rows[0], ["pc_perf_counter_s", "arduino_millis", "gsr_raw"])
        return status, rows[1:], serial.Serial, stderr.getvalue()

    def test_logging_preserves_each_reading_and_flushes_on_ctrl_c(self):
        status, rows, open_port, stderr = self.run_mock_logger([], [
            data(b"54787,648\r\n"),
            data(b"54911,"), b"", b"648\r\n",
            b"invalid\n",
            b"0" * MAX_LINE_BYTES, b"1,2\n",
            data(b"55036,647\n"),
            b"unfinished",
            KeyboardInterrupt(),
        ])
        open_port.assert_called_once_with(port="COM6", baudrate=9600, timeout=0.25, write_timeout=1.0)
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(rows, [
            ["10.0", "54787", "648"],
            ["10.125", "54911", "648"],
            ["10.5", "55036", "647"],
        ])

    def test_timing_warnings_leave_anomalous_samples_unchanged(self):
        status, rows, _, stderr = self.run_mock_logger([], [
            data(b"5000,648\n"), data(b"6500,648\n"), data(b"53,647\n"),
            data(b"53,647\n"), data(b"6750,648\n"), KeyboardInterrupt(),
        ], pc_times=[10.0, 10.001, 10.002, 10.003, 10.004], stable_at=(10.0, 5000))
        self.assertEqual(status, 0)
        self.assertEqual(rows, [
            ["10.0", "5000", "648"], ["10.001", "6500", "648"], ["10.002", "53", "647"],
            ["10.003", "53", "647"], ["10.004", "6750", "648"],
        ])
        self.assertEqual(stderr.count("Warning:"), 3)
        self.assertIn("Arduino timestamp gap of 1500 ms", stderr)
        self.assertIn("non-increasing Arduino timestamp (6500 -> 53 ms)", stderr)
        self.assertIn("Arduino 1.500 s, PC 0.001 s", stderr)

    def test_elapsed_disagreement_detects_compressed_stream_without_large_gaps(self):
        timing = TimingChecks(10.0, 5000)
        with redirect_stderr(io.StringIO()) as stderr:
            for i in range(1, 14):
                timing.observe(10.0 + 0.0002 * i, 5000 + 125 * i)
        self.assertEqual(stderr.getvalue().count("Warning:"), 1)
        self.assertIn("elapsed times disagree", stderr.getvalue())

    def test_duplicate_timestamp_is_flagged(self):
        timing = TimingChecks(10.0, 5000)
        with redirect_stderr(io.StringIO()) as stderr:
            timing.observe(10.125, 5000)
        self.assertIn("non-increasing Arduino timestamp (5000 -> 5000 ms)", stderr.getvalue())

    def test_disconnect_preserves_data_and_reports_port_error(self):
        status, rows, open_port, stderr = self.run_mock_logger(
            ["--port", "COM9", "--baud", "19200"],
            [data(b"54787,648\n"), SerialError("device disconnected")],
            stop_error=SerialError("port closed"),
        )
        open_port.assert_called_once_with(port="COM9", baudrate=19200, timeout=0.25, write_timeout=1.0)
        self.assertEqual(status, 1)
        self.assertEqual(rows, [["10.0", "54787", "648"]])
        self.assertIn("COM9: device disconnected", stderr)
        self.assertIn("could not send STOP", stderr)

    def test_requires_fresh_ack_and_rejects_other_sessions(self):
        status, rows, _, stderr = self.run_mock_logger([], [
            data(b"1,650\n", STALE),
            b"2,651\n",
            data(b"3,1024\n"),
            data(b"1125,648\n"),
            KeyboardInterrupt(),
        ], startup_lines=(b"ACK," + STALE + b"\n", data(b"0,600\n"), ACK), stable_at=(10.375, 1125))
        self.assertEqual(status, 0)
        self.assertEqual(rows, [["10.375", "1125", "648"]])
        self.assertIn("invalid GSR sample", stderr)

    def test_startup_burst_is_withheld_until_three_stable_intervals(self):
        chunks = [data(f"{1000 + i * 125},{640 + i}\n".encode()) for i in range(11)]
        pc_times = [10 + i * 0.001 for i in range(7)] + [10.125, 10.25, 10.375, 10.5]
        status, rows, _, stderr = self.run_mock_logger(
            [], [*chunks, KeyboardInterrupt()], pc_times=pc_times, stable_at=None, expected_discarded=9,
        )
        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(rows, [["10.375", "2125", "649"], ["10.5", "2250", "650"]])

    def test_unstable_interval_resets_the_stabilization_streak(self):
        chunks = [data(f"{1000 + i * 125},648\n".encode()) for i in range(7)]
        pc_times = [10.0, 10.125, 10.25, 10.251, 10.376, 10.501, 10.626]
        _, rows, _, _ = self.run_mock_logger(
            [], [*chunks, KeyboardInterrupt()], pc_times=pc_times, stable_at=None, expected_discarded=6,
        )
        self.assertEqual(rows, [["10.626", "1750", "648"]])

    def test_unstable_startup_times_out_without_opening_csv(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.read_until.side_effect = [ACK, data(b"1000,648\n"), data(b"1125,648\n")]
        serial = SimpleNamespace(Serial=MagicMock(return_value=connection), SerialException=SerialError)
        with (
            patch.dict("sys.modules", {"serial": serial}),
            patch("gsr_logger.uuid4", return_value=SimpleNamespace(hex=SESSION_HEX)),
            patch("gsr_logger.time.sleep"),
            patch("gsr_logger.time.monotonic", side_effect=[0, 0, 0, 0, 0, 10.1]),
            patch("gsr_logger.time.perf_counter", side_effect=[0, 1, 1.001]),
            patch.object(Path, "open") as open_csv,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(main([]), 1)
        open_csv.assert_not_called()
        self.assertEqual(connection.write.call_args_list, [call(START), call(STOP)])
        self.assertIn("Discarded 2 startup samples", stderr.getvalue())

    def test_handshake_retries_and_times_out_without_opening_csv(self):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.read_until.side_effect = [b"", b"ACK," + STALE + b"\n"]
        serial = SimpleNamespace(Serial=MagicMock(return_value=connection), SerialException=SerialError)
        with (
            patch.dict("sys.modules", {"serial": serial}),
            patch("gsr_logger.uuid4", return_value=SimpleNamespace(hex=SESSION_HEX)),
            patch("gsr_logger.time.sleep"),
            patch("gsr_logger.time.monotonic", side_effect=[0, 0, 1.1, 5.1]),
            patch.object(Path, "open") as open_csv,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(main([]), 1)
        open_csv.assert_not_called()
        self.assertEqual(connection.write.call_args_list, [call(START), call(START), call(STOP)])
        connection.__exit__.assert_called_once()
        self.assertIn("No fresh START acknowledgment", stderr.getvalue())

    def test_active_session_times_out_if_board_stops_sending(self):
        output = io.StringIO()
        with patch("gsr_logger.time.monotonic", side_effect=[0, 0, 5.1]):
            with self.assertRaisesRegex(TimeoutError, "No valid GSR samples"):
                log_samples(iter([None]), output, (10.0, 1000, 648))

    def test_unavailable_port_does_not_create_csv(self):
        serial = SimpleNamespace(
            Serial=MagicMock(side_effect=SerialError("port unavailable")),
            SerialException=SerialError,
        )
        with (
            patch.dict("sys.modules", {"serial": serial}),
            patch.object(Path, "open") as open_csv,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(main([]), 1)
        open_csv.assert_not_called()
        self.assertIn("COM6: port unavailable", stderr.getvalue())

    def test_missing_dependency_reports_install_command(self):
        with patch.dict("sys.modules", {"serial": None}), redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main([]), 1)
        self.assertIn("python -m pip install pyserial", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
