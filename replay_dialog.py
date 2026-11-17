"""Hardware-free replay controls for the experiment application."""

import csv
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from replay_session import load_replay


class ReplayDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Replay Session")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.session = None
        self.loading = False
        self.loaded = queue.SimpleQueue()
        self.status = tk.StringVar(value="Choose a baseline and two concurrent recordings.")
        self.progress = tk.DoubleVar()
        self.sources = []
        self.inputs = []
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        for row, (label, extension) in enumerate((
            ("Baseline profile", "json"), ("GSR recording", "csv"), ("Xbox recording", "csv"),
            ("GSR clock (optional override)", "json"), ("Xbox clock (optional override)", "json"),
            ("GSR interpretation rules (optional)", "json"),
            ("Joystick rules and context (optional)", "json"),
        )):
            variable = tk.StringVar()
            self.sources.append(variable)
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8))
            entry = ttk.Entry(frame, textvariable=variable, width=62)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            button = ttk.Button(
                frame, text="Browse", command=lambda v=variable, ext=extension: self.browse(v, ext)
            )
            button.grid(row=row, column=2, padx=(6, 0))
            self.inputs.extend((entry, button))

        controls = ttk.Frame(frame)
        controls.grid(row=7, columnspan=3, sticky="w", pady=8)
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause)
        self.resume_button = ttk.Button(controls, text="Resume", command=self.resume)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop)
        for button in (self.start_button, self.pause_button, self.resume_button, self.stop_button):
            button.pack(side="left", padx=(0, 6))
        ttk.Label(frame, textvariable=self.status, wraplength=680).grid(row=8, columnspan=3, sticky="w")
        ttk.Progressbar(frame, variable=self.progress, maximum=100).grid(row=9, columnspan=3, sticky="ew", pady=5)
        self.panels = []
        for row, title in enumerate(("GSR (raw ADC counts)", "Joystick behaviour"), start=10):
            panel = ttk.LabelFrame(frame, text=title, padding=6)
            panel.grid(row=row, columnspan=3, sticky="ew", pady=3)
            value = tk.StringVar(value="Waiting for replay.")
            self.panels.append(value)
            ttk.Label(panel, textvariable=value, font=("Consolas", 9), wraplength=680, justify="left").pack(anchor="w")
        interpretations = ttk.Frame(frame)
        interpretations.grid(row=12, columnspan=3, sticky="nsew")
        self.interpretations = []
        for column, source in enumerate(("GSR", "Joystick")):
            interpretations.columnconfigure(column, weight=1, uniform="interpretation")
            panel = ttk.LabelFrame(interpretations, text=f"{source} experimental interpretation", padding=6)
            panel.grid(row=0, column=column, sticky="nsew", padx=3)
            output = tk.Text(panel, height=8, width=38, wrap="word", font=("Consolas", 9), state="disabled")
            scroll = ttk.Scrollbar(panel, orient="vertical", command=output.yview)
            output.configure(yscrollcommand=scroll.set)
            output.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            self.interpretations.append(output)
        self.show_interpretations("Waiting for replay.", "Waiting for replay.")
        self.refresh()
        self.audit_location = tk.StringVar()
        ttk.Label(frame, textvariable=self.audit_location, wraplength=680).grid(row=13, columnspan=3, sticky="w")
        self.timer = self.after(100, self.poll)

    def show_interpretations(self, *messages):
        for output, message in zip(self.interpretations, messages):
            if output.get("1.0", "end-1c") != message:
                output.configure(state="normal")
                output.delete("1.0", "end")
                output.insert("1.0", message)
                output.configure(state="disabled")

    def browse(self, variable, extension):
        path = filedialog.askopenfilename(parent=self, filetypes=[(extension.upper(), f"*.{extension}")])
        if path:
            variable.set(path)

    def refresh(self):
        active = self.session is not None and not self.session.finished
        running = active and self.session.started_at is not None
        for widget in (*self.inputs, self.start_button):
            widget.configure(state="disabled" if active or self.loading else "normal")
        self.pause_button.configure(state="normal" if running else "disabled")
        self.resume_button.configure(state="normal" if active and not running else "disabled")
        self.stop_button.configure(state="normal" if active else "disabled")

    def start(self):
        names = [variable.get().strip() for variable in self.sources]
        if not all(names[:3]):
            self.status.set("Select the baseline, GSR recording, and Xbox recording.")
            return
        self.session = None
        self.loading = True
        self.progress.set(0)
        self.audit_location.set("")
        for panel in self.panels:
            panel.set("Waiting for replay.")
        self.show_interpretations("Waiting for replay.", "Waiting for replay.")
        self.status.set("Reading files and verifying shared clock.")
        self.refresh()

        def worker():
            try:
                self.loaded.put((load_replay(*(Path(name) if name else None for name in names), audit=True), None))
            except (OSError, ValueError, csv.Error) as error:
                self.loaded.put((None, str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def pause(self):
        self.session.pause(time.monotonic())
        self.refresh()

    def resume(self):
        self.session.resume(time.monotonic())
        self.refresh()

    def stop(self):
        self.session = None
        self.status.set("Replay stopped.")
        self.progress.set(0)
        for panel in self.panels:
            panel.set("Replay stopped.")
        self.show_interpretations("Replay stopped.", "Replay stopped.")
        self.refresh()

    def poll(self):
        try:
            loaded, error = self.loaded.get_nowait()
        except queue.Empty:
            pass
        else:
            self.loading = False
            if error is not None:
                self.status.set(f"Cannot replay: {error}")
            else:
                self.session = loaded
                self.audit_location.set(f"GSR audit: {loaded.audit_path}\nJoystick audit: {loaded.joystick_audit_path}")
                loaded.resume(time.monotonic())
            self.refresh()
        if self.session is not None:
            session = self.session
            for panel, result in zip(self.panels, session.tick(time.monotonic())):
                panel.set(result)
            self.show_interpretations(session.interpretation_text(), session.joystick_interpretation_text())
            self.progress.set(100 * session.position / session.duration)
            state = "Finished" if session.finished else "Playing" if session.started_at is not None else "Paused"
            self.status.set(
                f"{state} · {session.baseline.participant_id} · "
                f"{session.position:.1f} / {session.duration:.1f} s · PC {session.first_pc + session.position:.3f} s"
            )
            if session.finished:
                self.refresh()
        self.timer = self.after(100, self.poll)

    def close(self):
        self.after_cancel(self.timer)
        self.destroy()
