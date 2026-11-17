"""Session state, participant baselines, and offline task comparison."""

import csv
import filecmp
import json
import math
import re
import shutil
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import features
import gsr_features
from fuse_features import FusedWindow, STEP_SECONDS, WINDOW_SECONDS, fuse_recordings
from recording_clock import metadata_path, verify_shared_clock


DATA_DIR = Path(__file__).resolve().parent / "data"
PARTICIPANTS_DIR = DATA_DIR / "participants"
BASELINE_FILENAME = "baseline_profile.json"
MIN_BASELINE_WINDOWS = 3
MAD_SCALE = 1.4826
METRICS = (
    "movement_variability", "reversals", "reversals_per_active_minute",
    "input_roughness_per_s", "idle_seconds", "idle_percent", "mean_gsr_raw",
)
COMPARISONS = ("difference_from_mean", "z_score", "robust_z_score")


class Stage(Enum):
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    STOPPING = auto()
    AWAITING_XBOX = auto()
    ANALYZING = auto()


TRANSITIONS = {
    Stage.IDLE: {Stage.STARTING},
    Stage.STARTING: {Stage.RECORDING, Stage.STOPPING, Stage.IDLE},
    Stage.RECORDING: {Stage.STOPPING, Stage.IDLE},
    Stage.STOPPING: {Stage.AWAITING_XBOX, Stage.IDLE},
    Stage.AWAITING_XBOX: {Stage.ANALYZING, Stage.IDLE},
    Stage.ANALYZING: {Stage.AWAITING_XBOX, Stage.IDLE},
}


@dataclass
class Workflow:
    stage: Stage = Stage.IDLE

    def move(self, target: Stage) -> None:
        if target not in TRANSITIONS[self.stage]:
            raise ValueError(f"Cannot move from {self.stage.name} to {target.name}.")
        self.stage = target

    def start(self, kind: str, has_baseline: bool) -> None:
        if kind not in ("baseline", "task"):
            raise ValueError("Unknown experiment phase.")
        if kind == "task" and not has_baseline:
            raise ValueError("Save a valid participant baseline before starting a task.")
        self.move(Stage.STARTING)


def participant_directory(participant: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", participant):
        raise ValueError("Use 1–32 letters, digits, hyphens, or underscores, starting with a letter or digit.")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]", participant, re.IGNORECASE):
        raise ValueError("Choose a participant ID that is not a reserved Windows name.")
    return PARTICIPANTS_DIR / participant.upper()


@dataclass(frozen=True)
class Session:
    participant: str
    kind: str
    directory: Path

    @classmethod
    def create(cls, participant: str, kind: str) -> "Session":
        folder = participant_directory(participant)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = folder / f"{kind}_{stamp}"
        directory.mkdir(parents=True)
        return cls(folder.name, kind, directory)


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float | None
    median: float | None
    stddev: float | None
    mad: float | None
    iqr: float | None


@dataclass(frozen=True)
class Baseline:
    participant_id: str
    session: str
    window_count: int
    metrics: dict[str, Summary]
    window_seconds: int = WINDOW_SECONDS
    step_seconds: int = STEP_SECONDS
    task: str = "standardized familiar teleoperation"
    schema_version: int = 1


def summarize(values: Sequence[float]) -> Summary:
    if not values:
        return Summary(0, None, None, None, None, None)
    center = statistics.median(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else None
    return Summary(
        len(values), statistics.mean(values), center, statistics.pstdev(values),
        statistics.median(abs(value - center) for value in values),
        quartiles[2] - quartiles[0] if quartiles else None,
    )


def create_baseline(session: Session, windows: Sequence[FusedWindow]) -> Baseline:
    if len(windows) < MIN_BASELINE_WINDOWS:
        raise ValueError(f"Baseline needs at least {MIN_BASELINE_WINDOWS} valid fused windows (40 s minimum overlap).")
    summaries = {}
    for metric in METRICS:
        values = [getattr(window, metric) for window in windows]
        summaries[metric] = summarize([value for value in values if value is not None and math.isfinite(value)])
    return Baseline(session.participant, session.directory.name, len(windows), summaries)


def deviations(value: float | None, baseline: Summary) -> tuple[float | None, float | None, float | None]:
    if value is None or not math.isfinite(value) or baseline.mean is None:
        return None, None, None
    difference = value - baseline.mean
    if baseline.count < MIN_BASELINE_WINDOWS:
        return difference, None, None
    z_score = difference / baseline.stddev if baseline.stddev else None
    robust_score = (value - baseline.median) / (MAD_SCALE * baseline.mad) if baseline.mad else None
    return difference, z_score, robust_score


def compare_windows(windows: Sequence[FusedWindow], baseline: Baseline):
    for window in windows:
        scores = (
            score
            for metric in METRICS
            for score in deviations(getattr(window, metric), baseline.metrics[metric])
        )
        yield (*window, baseline.session, *scores)


def latest_baseline(participant: str) -> tuple[Path, Baseline] | None:
    folder = participant_directory(participant)
    paths = sorted(folder.glob(f"baseline_*/{BASELINE_FILENAME}"))
    if not paths:
        return None
    path = paths[-1]
    return path, load_baseline(path, folder.name)


def load_baseline(path: Path, participant: str | None = None) -> Baseline:
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        data["metrics"] = {key: Summary(**data["metrics"][key]) for key in METRICS}
        baseline = Baseline(**data)
        if (
            (participant is not None and (baseline.participant_id != participant or baseline.session != path.parent.name))
            or baseline.schema_version != 1 or baseline.window_count < MIN_BASELINE_WINDOWS
            or (baseline.window_seconds, baseline.step_seconds) != (WINDOW_SECONDS, STEP_SECONDS)
        ):
            raise ValueError("profile metadata does not match this participant or analysis")
        for summary in baseline.metrics.values():
            if not isinstance(summary.count, int) or not 0 <= summary.count <= baseline.window_count:
                raise ValueError("invalid baseline sample count")
            if summary.count:
                if not all(math.isfinite(value) for value in (summary.mean, summary.median, summary.stddev, summary.mad)):
                    raise ValueError("invalid baseline feature summary")
                if summary.stddev < 0 or summary.mad < 0:
                    raise ValueError("invalid baseline scale")
            elif any(value is not None for value in (summary.mean, summary.median, summary.stddev, summary.mad, summary.iqr)):
                raise ValueError("invalid empty baseline feature")
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Cannot load baseline {path}: {error}") from error
    return baseline


def write_csv(path: Path, columns, rows) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


def write_baseline(path: Path, baseline: Baseline) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(asdict(baseline), stream, indent=2, allow_nan=False)
    temporary.replace(path)


def archive_xbox(source: Path, session: Session) -> Path:
    folder = session.directory / "xbox"
    folder.mkdir(exist_ok=True)
    destination = folder / source.name
    for original, target in ((source, destination), (metadata_path(source), metadata_path(destination))):
        if target.exists():
            if not filecmp.cmp(original, target, shallow=False):
                raise ValueError(f"A different copy of {original.name} is already in this session.")
        else:
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(original, temporary)
            if not filecmp.cmp(original, temporary, shallow=False):
                raise ValueError(f"{original.name} changed while copying. Stop Xbox capture with B first.")
            temporary.replace(target)
    return destination


def analyze_session(
    session: Session, xbox_source: Path, baseline: Baseline | None, warn: Callable[[str], None],
) -> Path:
    gsr_files = list(session.directory.glob("gsr_*.csv"))
    if len(gsr_files) != 1:
        raise ValueError("Expected one GSR recording in the session directory.")
    gsr = gsr_files[0]
    verify_shared_clock(xbox_source, gsr)
    xbox = archive_xbox(xbox_source, session)
    verify_shared_clock(xbox, gsr)
    windows = fuse_recordings(features.read_recording(xbox), gsr_features.read_recording(gsr), warn)
    write_csv(session.directory / "fused_features.csv", FusedWindow._fields, windows)
    if session.kind == "baseline":
        result = session.directory / BASELINE_FILENAME
        write_baseline(result, create_baseline(session, windows))
    else:
        if baseline is None or baseline.participant_id != session.participant:
            raise ValueError("A baseline for this participant is required.")
        result = session.directory / "task_comparison.csv"
        columns = [f"{metric}_{comparison}" for metric in METRICS for comparison in COMPARISONS]
        write_baseline(session.directory / "baseline_used.json", baseline)
        write_csv(result, (*FusedWindow._fields, "baseline_session", *columns), compare_windows(windows, baseline))
    return result
