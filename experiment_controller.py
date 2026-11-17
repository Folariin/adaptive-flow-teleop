"""Tkinter controller for GSR capture and manually operated Xbox sessions."""

import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from experiment_session import (
    DATA_DIR, Session, Stage, Workflow, analyze_session, latest_baseline, participant_directory,
)
from live_features import monitor_task
from replay_dialog import ReplayDialog


ROOT = Path(__file__).resolve().parent
STOP_TIMEOUT_SECONDS = 8


class ExperimentController:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.workflow = Workflow()
        self.session = None
        self.process = None
        self.baseline = None
        self.result = None
        self.started_at = None
        self.stop_deadline = None
        self.forced_stop = False
        self.closing = False
        self.events = queue.Queue()
        self.live_stop = None
        self.live_updates = queue.SimpleQueue()
        self.environment_error = self.capture_environment_error()

        self.participant = tk.StringVar()
        self.location = tk.StringVar()
        self.readiness = tk.StringVar()
        self.status = tk.StringVar(value="Enter a pseudonymous participant ID to begin.")
        self.elapsed = tk.StringVar(value="Elapsed: 0.0 s")
        self.files = tk.StringVar()
        self.gsr_live = tk.StringVar(value="Available during a task.")
        self.xbox_live = tk.StringVar(value="Available during a task.")

        root.title("Adaptive-Flow Task Structuring in Teleoperation")
        root.minsize(670, 430)
        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Experiment controller", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        identity = ttk.Frame(frame)
        identity.pack(fill="x", pady=(12, 4))
        ttk.Label(identity, text="Participant ID").pack(side="left")
        self.entry = ttk.Entry(identity, textvariable=self.participant, width=18)
        self.entry.pack(side="left", padx=8)
        self.entry.focus_set()
        for variable in (self.location, self.readiness):
            ttk.Label(frame, textvariable=variable, wraplength=625).pack(anchor="w", pady=2)
        ttk.Separator(frame).pack(fill="x", pady=10)
        ttk.Label(
            frame, wraplength=625, justify="left",
            text=(
                "Manual Xbox / robot workflow\n"
                "After GSR is ready, run the copied Xbox command for a task, or python xbox_controller.py for baseline. "
                "Press B in the Xbox recording to stop it before using Stop in this app. "
                "Then select its closed CSV. Use the standardized familiar teleoperation task for baseline."
            ),
        ).pack(anchor="w", pady=(0, 10))

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=(0, 8))
        self.start_baseline = ttk.Button(buttons, text="Start Baseline", command=lambda: self.start("baseline"))
        self.stop_baseline = ttk.Button(buttons, text="Stop & Save Baseline", command=self.stop)
        self.start_task = ttk.Button(buttons, text="Start Task", command=lambda: self.start("task"))
        self.stop_task = ttk.Button(buttons, text="Stop & Analyze Task", command=self.stop)
        for button in (self.start_baseline, self.stop_baseline, self.start_task, self.stop_task):
            button.pack(side="left", padx=(0, 6))

        more = ttk.Frame(frame)
        more.pack(fill="x", pady=(0, 10))
        self.choose = ttk.Button(more, text="Select Xbox CSV", command=self.select_xbox)
        self.abandon = ttk.Button(more, text="End Without Analysis", command=self.abandon_session)
        self.view = ttk.Button(more, text="View Results", command=self.view_results)
        self.copy_xbox = ttk.Button(more, text="Copy Task Xbox Command", command=self.copy_xbox_command)
        self.replay = ttk.Button(more, text="Replay Session", command=lambda: ReplayDialog(self.root))
        for button in (self.choose, self.abandon, self.view, self.copy_xbox, self.replay):
            button.pack(side="left", padx=(0, 6))

        ttk.Label(frame, textvariable=self.elapsed).pack(anchor="w")
        ttk.Label(frame, textvariable=self.status, wraplength=625, justify="left").pack(anchor="w", pady=4)
        ttk.Label(frame, textvariable=self.files, wraplength=625, justify="left").pack(anchor="w", pady=4)
        for label, variable in (("GSR (raw ADC counts)", self.gsr_live), ("Joystick behaviour", self.xbox_live)):
            panel = ttk.LabelFrame(frame, text=label, padding=6)
            panel.pack(fill="x", pady=3)
            ttk.Label(panel, textvariable=variable, font=("Consolas", 9), wraplength=625, justify="left").pack(anchor="w")
        ttk.Label(
            frame, text="Live values and baseline deviations are descriptive. A dash means unavailable.",
            wraplength=625,
        ).pack(anchor="w", pady=3)
        self.participant.trace_add("write", self.refresh_participant)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_participant()
        root.after(200, self.poll)

    @staticmethod
    def capture_environment_error() -> str | None:
        if sys.platform != "win32" or sys.implementation.name != "cpython" or sys.version_info < (3, 10):
            return "Capture requires Windows with CPython 3.10 or later for clock verification."
        if importlib.util.find_spec("serial") is None:
            return "GSR requires pySerial. Install it with python -m pip install pyserial."
        return None

    def refresh_participant(self, *_args):
        self.baseline = None
        self.result = None
        self.participant_folder = None
        self.files.set("")
        try:
            participant = self.participant.get().strip()
            if not participant:
                self.location.set("Participant directory: enter an ID")
                return
            self.participant_folder = participant_directory(participant)
            self.location.set(f"Participant directory: {self.participant_folder}")
            self.baseline = latest_baseline(participant)
            if self.baseline is not None:
                self.result = self.baseline[0]
                self.files.set(f"Active baseline: {self.result}")
            tasks = sorted(self.participant_folder.glob("task_*/task_comparison.csv"))
            if tasks and (self.result is None or tasks[-1].parent.name[5:] > self.result.parent.name[9:]):
                self.result = tasks[-1]
                self.files.set(f"Latest result: {self.result}\nActive baseline: {self.baseline[0] if self.baseline else 'none'}")
        except (OSError, ValueError) as error:
            if self.participant_folder is None:
                self.location.set(str(error))
            else:
                self.files.set(str(error))
        finally:
            self.refresh_controls()

    def refresh_controls(self):
        stage = self.workflow.stage
        idle = stage is Stage.IDLE
        available = idle and self.participant_folder is not None and not self.environment_error
        stopping_allowed = stage in (Stage.STARTING, Stage.RECORDING)
        kind = self.session.kind if self.session else None
        buttons = (
            (self.entry, idle),
            (self.start_baseline, available),
            (self.start_task, available and self.baseline is not None),
            (self.stop_baseline, stopping_allowed and kind == "baseline"),
            (self.stop_task, stopping_allowed and kind == "task"),
            (self.choose, stage is Stage.AWAITING_XBOX),
            (self.abandon, stage is Stage.AWAITING_XBOX),
            (self.view, idle and self.result is not None),
            (self.copy_xbox, stage is Stage.RECORDING and kind == "task"),
            (self.replay, idle),
        )
        for widget, enabled in buttons:
            widget.configure(state="normal" if enabled and not self.closing else "disabled")
        gsr = {
            Stage.IDLE: "GSR COM6: ready to check when capture starts",
            Stage.STARTING: "GSR COM6: waiting for handshake and stable samples",
            Stage.RECORDING: "GSR COM6: stable samples, recording",
            Stage.STOPPING: "GSR COM6: stopping",
            Stage.AWAITING_XBOX: "GSR COM6: stopped",
            Stage.ANALYZING: "GSR COM6: stopped",
        }[stage]
        baseline = "baseline saved" if self.baseline else "no valid baseline selected"
        self.readiness.set(f"{self.environment_error or gsr} | Xbox: manual | {baseline}")

    def start(self, kind):
        try:
            self.workflow.start(kind, self.baseline is not None)
            self.session = Session.create(self.participant.get().strip(), kind)
            if kind == "task":
                (self.session.directory / "xbox_live").mkdir()
            log_path = self.session.directory / "gsr_process.log"
            with log_path.open("xb") as log:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "gsr_logger.py"), "--output-dir",
                     str(self.session.directory), "--stop-on-stdin"],
                    cwd=ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except (OSError, ValueError) as error:
            if self.workflow.stage is Stage.STARTING and self.process is None:
                self.workflow.move(Stage.IDLE)
            self.status.set(f"Could not start capture: {error}")
            self.refresh_controls()
            return
        self.started_at = None
        self.forced_stop = False
        self.elapsed.set("Elapsed: 0.0 s")
        self.files.set(f"Session: {self.session.directory}\nGSR log: {log_path}")
        self.status.set("Starting GSR. Wait for stable capture before starting Xbox manually.")
        self.gsr_live.set("Waiting for capture." if kind == "task" else "Available during a task.")
        self.xbox_live.set("Waiting for capture." if kind == "task" else "Available during a task.")
        if kind == "task":
            self.live_stop = threading.Event()
            updates = self.live_updates = queue.SimpleQueue()
            threading.Thread(
                target=monitor_task,
                args=(self.session.directory, self.baseline[1], self.live_stop, updates.put),
                daemon=True,
            ).start()
        self.refresh_controls()

    def copy_xbox_command(self):
        command = subprocess.list2cmdline(
            ["python", str(ROOT / "xbox_controller.py"), "--live-output-dir", str(self.session.directory / "xbox_live")]
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self.status.set("Xbox command copied. Run it manually in a terminal with your LeRobot environment.")

    def stop(self):
        if self.workflow.stage not in (Stage.STARTING, Stage.RECORDING):
            return
        self.workflow.move(Stage.STOPPING)
        if self.live_stop is not None:
            self.live_stop.set()
        self.stop_deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
        self.status.set("Stopping GSR and waiting for its CSV and clock sidecar to close.")
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.refresh_controls()

    def capture_finished(self, code):
        if self.live_stop is not None:
            self.live_stop.set()
            self.live_stop = None
        if self.session.kind == "task":
            self.gsr_live.set("Live capture ended.")
            self.xbox_live.set("Live capture ended.")
        was_stopping = self.workflow.stage is Stage.STOPPING
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process = None
        self.stop_deadline = None
        if self.closing:
            return
        recordings = list(self.session.directory.glob("gsr_*.csv"))
        if was_stopping and code == 0 and not self.forced_stop and len(recordings) == 1:
            self.workflow.move(Stage.AWAITING_XBOX)
            self.status.set("GSR saved. Stop Xbox with B if needed, then select its closed CSV.")
            self.root.after_idle(self.select_xbox)
        else:
            self.workflow.move(Stage.IDLE)
            if code == 0 and was_stopping and not self.forced_stop:
                self.status.set("Capture stopped before a GSR recording was ready.")
            elif self.forced_stop:
                self.status.set("GSR did not stop cooperatively. The process was terminated and this session was not analyzed.")
            else:
                log = self.session.directory / "gsr_process.log"
                try:
                    with log.open("rb") as stream:
                        stream.seek(max(0, log.stat().st_size - 2048))
                        tail = stream.read().decode(errors="replace").splitlines()
                    detail = tail[-1] if tail else f"exit code {code}"
                except OSError:
                    detail = f"exit code {code}"
                self.status.set(f"GSR capture ended unexpectedly. {detail}\nIf Xbox is running, stop it with B.")
        self.refresh_controls()

    def select_xbox(self):
        if self.closing or self.workflow.stage is not Stage.AWAITING_XBOX:
            return
        source = filedialog.askopenfilename(
            parent=self.root, initialdir=self.session.directory / "xbox_live" if self.session.kind == "task" else DATA_DIR,
            title="Select the Xbox CSV after stopping Xbox with B", filetypes=[("CSV recording", "*.csv")],
        )
        if not source:
            return
        self.workflow.move(Stage.ANALYZING)
        self.status.set("Validating shared clocks and sample coverage, then analyzing windows.")
        self.refresh_controls()
        baseline = self.baseline[1] if self.baseline else None

        def worker():
            warnings = []
            try:
                path = analyze_session(self.session, Path(source), baseline, warnings.append)
            except Exception as error:
                self.events.put((None, str(error), warnings))
            else:
                self.events.put((path, None, warnings))

        threading.Thread(target=worker, daemon=True).start()

    def analysis_finished(self, path, error, warnings):
        notes = "\n" + "\n".join(warnings) if warnings else ""
        if error is not None:
            self.workflow.move(Stage.AWAITING_XBOX)
            self.status.set(f"Analysis failed: {error}{notes}")
        else:
            self.workflow.move(Stage.IDLE)
            self.refresh_participant()
            self.result = path
            self.files.set(f"Result: {path}\nSession: {self.session.directory}")
            self.status.set(f"{'Baseline saved' if self.session.kind == 'baseline' else 'Task analyzed'}.{notes}")
        self.refresh_controls()

    def abandon_session(self):
        self.workflow.move(Stage.IDLE)
        self.status.set("Session ended without analysis. Captured files remain in the session directory.")
        self.refresh_controls()

    def view_results(self):
        if self.result is not None:
            try:
                os.startfile(self.result)
            except (AttributeError, OSError) as error:
                messagebox.showerror("Could not open result", f"{self.result}\n{error}", parent=self.root)

    def close(self):
        if self.process is not None:
            self.closing = True
            self.stop()
            self.refresh_controls()
        elif self.workflow.stage is Stage.ANALYZING:
            self.closing = True
            self.refresh_controls()
        else:
            self.root.destroy()

    def poll(self):
        if self.process is not None:
            now = time.monotonic()
            code = self.process.poll()
            if code is not None:
                self.capture_finished(code)
            elif self.stop_deadline is not None and now >= self.stop_deadline:
                self.forced_stop = True
                self.stop_deadline = None
                try:
                    self.process.terminate()
                except OSError as error:
                    self.status.set(f"Could not terminate the GSR process: {error}")
            elif self.workflow.stage is Stage.STARTING and any(self.session.directory.glob("gsr_*.csv")):
                self.started_at = now
                self.workflow.move(Stage.RECORDING)
                self.status.set(
                    "GSR is recording. Copy and manually run the task Xbox command."
                    if self.session.kind == "task" else "GSR is recording. The operator can now start Xbox manually."
                )
                self.refresh_controls()
            if self.started_at is not None and self.workflow.stage is Stage.RECORDING:
                self.elapsed.set(f"Elapsed: {now - self.started_at:.1f} s")
        try:
            result = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.analysis_finished(*result)
        while True:
            try:
                live = self.live_updates.get_nowait()
            except queue.Empty:
                break
            if self.workflow.stage in (Stage.STARTING, Stage.RECORDING):
                self.gsr_live.set(live["gsr"])
                self.xbox_live.set(live["xbox"])
        if self.closing and self.process is None and self.workflow.stage is not Stage.ANALYZING:
            self.root.destroy()
        else:
            self.root.after(200, self.poll)


def main():
    root = tk.Tk()
    ExperimentController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
