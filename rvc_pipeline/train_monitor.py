"""Monitor trwającego treningu Applio: postęp, szacowany czas, straty, stop, auto-kopia G/D.

Źródła danych (Applio 3.6.5, ``rvc/train/train.py``):
  * linia na koniec epoki:
    ``<model> | epoch=38 | step=1558 | time=10:12:45 | training_speed=0:01:11``,
  * pasek tqdm w trakcie epoki (``15/41 [00:25<00:44, …]``) - postęp w epoce,
  * straty w TensorBoard ``logs/<model>/eval/`` (``loss_avg_50/g/*`` co 50
    kroków) - czytane przez :mod:`rvc_pipeline.tb_scalars`,
  * zapis G/D następuje **po** wypisaniu linii epoki podzielnej przez
    ``--save-every-epoch``.
"""

from __future__ import annotations

import html
import math
import os
import re
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from rvc_pipeline.checkpoints import (
    checkpoint_drive_name,
    copy_atomic,
    latest_training_checkpoint,
    prune_checkpoint_pairs,
    wait_for_checkpoint_written,
)
from rvc_pipeline import save_log
from rvc_pipeline.tb_scalars import ScalarReader

EPOCH_LINE_RE = re.compile(
    r"epoch=(\d+)\s*\|\s*step=(\d+)\s*\|\s*time=[\d:]+\s*\|\s*"
    r"training_speed=(\d+):(\d{2}):(\d{2})"
)
_TQDM_RE = re.compile(r"(\d+)/(\d+)\s*\[")

LOSS_SERIES = {
    "loss_avg_50/g/total": "Generator: strata łączna",
    "loss_avg_50/g/mel": "Mel: różnica widma",
}

# Ile ostatnich epok bierze mediana tempa: pierwsze epoki są ~2× wolniejsze
# (rozgrzewka GPU, cache danych - pomiar w Colabie T4, dziennik Krok 31).
_PACE_WINDOW = 5
AUTO_BACKUP_KEEP = 2


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


@dataclass
class EpochSample:
    epoch: int
    step: int
    seconds: int


class TrainingStopped(Exception):
    """Trening przerwany przyciskiem „Zatrzymaj trening”."""


class TrainingMonitor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_dir: Path | None = None
        self.total_epochs = 0
        self.save_every = 0
        self.backup_dir: Path | None = None
        self.phase = "Przygotowanie…"
        self.training_started = False

        self.epochs: list[EpochSample] = []
        self.in_epoch: tuple[int, int] | None = None
        self.last_ckpt_epoch: int | None = None
        self.backup_status = ""

        self.stop_requested = False
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._save_ts: float | None = None
        self._saving_epoch: int | None = None
        self._backup_threads: list[threading.Thread] = []
        self.last_log_at = time.time()

        self._loss_reader: ScalarReader | None = None
        self._loss_points: list[tuple[str, int, float]] = []
        self.loss_version = 0

    # --- konfiguracja / proces --------------------------------------------

    def configure(
        self,
        model_dir: Path,
        total_epochs: int,
        save_every: int,
        backup_dir: Path | None,
        resume_epoch: int | None,
    ) -> None:
        self.model_dir = model_dir
        self.total_epochs = total_epochs
        self.save_every = save_every
        self.backup_dir = backup_dir
        self.last_ckpt_epoch = resume_epoch
        self._loss_reader = ScalarReader(model_dir / "eval", set(LOSS_SERIES))

    def attach(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
        if self.stop_requested:
            _kill_tree(proc.pid)

    def detach(self) -> None:
        with self._lock:
            self._proc = None

    # --- log ---------------------------------------------------------------

    def feed(self, line: str) -> bool:
        """Przetwarza linię logu; True, gdy zakończyła się epoka."""
        self.last_log_at = time.time()
        m = EPOCH_LINE_RE.search(line)
        if m:
            epoch, step = int(m.group(1)), int(m.group(2))
            seconds = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
            self.epochs.append(EpochSample(epoch, step, seconds))
            self.in_epoch = None
            if self.save_every and epoch % self.save_every == 0:
                self._on_checkpoint_saving(epoch)
            self._read_losses()
            return True
        t = _TQDM_RE.search(line)
        if t and self.training_started:
            # Applio zapisuje G/D synchronicznie po linii epoki - pasek kolejnej
            # epoki oznacza, że zapis już się skończył.
            self._saving_epoch = None
            done, total = int(t.group(1)), int(t.group(2))
            if 0 < total and done <= total:
                self.in_epoch = (done, total)
        return False

    def _read_losses(self) -> None:
        if self._loss_reader is None:
            return
        new = self._loss_reader.read_new()
        if new:
            self._loss_points.extend(new)
            self.loss_version += 1

    @property
    def steps_per_epoch(self) -> int | None:
        for s in reversed(self.epochs):
            if s.epoch > 0 and s.step > 0:
                return max(1, round(s.step / s.epoch))
        if self.in_epoch:
            return self.in_epoch[1]
        return None

    def loss_rows(self) -> list[dict]:
        spe = self.steps_per_epoch
        rows = []
        for tag, step, value in self._loss_points:
            x = round(step / spe, 2) if spe else step
            rows.append({"epoka": x, "strata": round(value, 4), "seria": LOSS_SERIES[tag]})
        return rows

    def loss_chart_html(self) -> str:
        return loss_chart_svg(self.loss_rows())

    # --- postęp ------------------------------------------------------------

    def pace_seconds(self) -> float | None:
        recent = [s.seconds for s in self.epochs[-_PACE_WINDOW:] if s.seconds > 0]
        return float(statistics.median(recent)) if recent else None

    def progress_html(self) -> str:
        total = self.total_epochs
        if not total or not (self.epochs or self.in_epoch):
            return _card(0.0, html.escape(self.phase), [])

        current = self.epochs[-1].epoch if self.epochs else (self.last_ckpt_epoch or 0)
        frac = 0.0
        if self.in_epoch and current < total:
            frac = self.in_epoch[0] / self.in_epoch[1]
        pct = min(1.0, (current + frac) / total)

        details = []
        silent_for = time.time() - self.last_log_at
        if silent_for >= 45 and self.training_started:
            details.append(
                f"<b>Brak nowych linii w logu od {int(silent_for)} s</b> - "
                "jeśli trwa &gt; 5-10 min przy tym samym kroku, to raczej zawieszenie "
                "DataLoadera/GPU: użyj „Zatrzymaj trening” albo Interrupt w Colabie."
            )
        pace = self.pace_seconds()
        if pace:
            remaining = max(0.0, (total - current - frac) * pace)
            details.append(
                f"Tempo: <b>{format_duration(pace)}</b>/epokę"
            )
            if current < total:
                details.append(f"Zostało ok. <b>{format_duration(remaining)}</b>")
            if len(self.epochs) < _PACE_WINDOW:
                details.append(
                    "Pierwsze epoki są wolniejsze - szacunek "
                    "zwykle spada po kilku-kilkunastu epokach."
                )
        if self.last_ckpt_epoch:
            details.append(f"Ostatni checkpoint G/D: epoka <b>{self.last_ckpt_epoch}</b>")
        if self.backup_status:
            details.append(html.escape(self.backup_status))
        title = f"Epoka <b>{current}</b> / {total}"
        if self.in_epoch and current < total:
            title += f" · krok {self.in_epoch[0]}/{self.in_epoch[1]} w epoce {current + 1}"
        return _card(pct, title, details)

    # --- checkpoint / auto-kopia ------------------------------------------

    @property
    def checkpoint_save_pending(self) -> bool:
        """Applio zapisuje teraz G/D - zabicie procesu uszkodziłoby checkpoint."""
        return self._saving_epoch is not None

    def _on_checkpoint_saving(self, epoch: int) -> None:
        ts = time.time()
        self._save_ts = ts
        self._saving_epoch = epoch
        thread = threading.Thread(
            target=self._after_checkpoint, args=(epoch, ts), daemon=True
        )
        self._backup_threads.append(thread)
        thread.start()

    def _after_checkpoint(self, epoch: int, ts: float) -> None:
        assert self.model_dir is not None
        written = wait_for_checkpoint_written(self.model_dir, ts)
        if self._saving_epoch == epoch:
            self._saving_epoch = None
        if self._save_ts != ts:
            # --save-only-latest: pliki należą już do nowszej epoki, jej wątek je skopiuje.
            return
        if not written:
            self.backup_status = f"Checkpoint z epoki {epoch}: nie doczekano się zapisu G/D."
            save_log.record(self.backup_status)
            return
        self.last_ckpt_epoch = max(epoch, self.last_ckpt_epoch or 0)
        if self.backup_dir is None:
            save_log.record(f"Checkpoint G/D z epoki {epoch} zapisany lokalnie w {self.model_dir}")
            return
        self.backup_status = f"Kopiuję checkpoint G/D z epoki {epoch} na Dysk Google…"
        try:
            for kind in ("G", "D"):
                src = latest_training_checkpoint(self.model_dir, kind)
                if src is None:
                    raise OSError(f"brak pliku {kind}_*.pth")
                dest = self.backup_dir / checkpoint_drive_name(kind, self.model_name, epoch)
                copy_atomic(src, dest)
            prune_checkpoint_pairs(self.backup_dir, AUTO_BACKUP_KEEP)
        except OSError as exc:
            self.backup_status = f"Kopia checkpointu G/D z epoki {epoch} na Dysk nie powiodła się: {exc}"
            save_log.record(self.backup_status)
            return
        self.backup_status = (
            f"Checkpoint G/D z epoki {epoch} jest na Dysku Google, "
            f"zapis o {save_log.now_hm('%H:%M')}."
        )
        save_log.record(
            f"Checkpoint G/D z epoki {epoch} zapisany na Dysku: {self.backup_dir}. "
            f"Zostają {AUTO_BACKUP_KEEP} ostatnie pary."
        )

    def backups_pending(self) -> bool:
        return any(t.is_alive() for t in self._backup_threads)

    # --- stop --------------------------------------------------------------

    def request_stop(self) -> str:
        """Zgłasza zatrzymanie i wraca od razu (handler przycisku nie może wisieć).

        Proces zabija pętla logu (``_stream_step``) - po dokończeniu ewentualnego
        zapisu G/D - i dalej pakuje model z ostatniego checkpointu. Wątek
        zapasowy dobija proces, gdyby pętla logu już nie działała.
        """
        first = not self.stop_requested
        self.stop_requested = True
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return "Zatrzymanie zgłoszone - trening przerwie się przed kolejnym krokiem."
        if first:
            threading.Thread(
                target=self._fallback_kill, args=(proc,), daemon=True
            ).start()
        if self.checkpoint_save_pending:
            return (
                "Zatrzymuję: Applio kończy zapis checkpointu G/D, potem trening "
                "zostanie przerwany, a model spakowany do ZIP."
            )
        return (
            "Zatrzymuję trening - za chwilę po prawej pojawi się ZIP z modelem "
            "z ostatniego checkpointu."
        )

    def _fallback_kill(self, proc: subprocess.Popen, grace_s: float = 45.0) -> None:
        deadline = time.time() + 240
        while self.checkpoint_save_pending and time.time() < deadline:
            time.sleep(1)
        end = time.time() + grace_s
        while time.time() < end:
            if proc.poll() is not None:
                return
            time.sleep(1)
        _kill_tree(proc.pid)


_CHART_COLORS = ("#1d4ed8", "#c2410c")
_CHART_W, _CHART_H = 640, 180
_CHART_PAD_L, _CHART_PAD_R, _CHART_PAD_T, _CHART_PAD_B = 48, 14, 10, 26
_CHART_MAX_POINTS = 400


def _nice_num(value: float) -> str:
    return f"{value:.3g}"


def _nice_step(span: float, count: int = 4) -> float:
    raw = span / count
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _moving_average(values: list[float], window: int) -> list[float]:
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _loss_panel(name: str, pts: list[tuple[float, float]], color: str, uid: int) -> str:
    """Jeden panel = jedna seria z własną skalą Y (wspólna oś spłaszczała obie)."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    smooth = _moving_average(ys, max(3, len(ys) // 12))

    x0, x1 = min(xs), max(xs)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    # Pierwsze punkty (rozgrzewka) bywają dużo wyżej - nie rozciągają osi,
    # są tylko przycięte u góry panelu.
    skip = max(1, len(ys) // 8) if len(ys) >= 8 else 0
    y0 = min(ys)
    y1 = max(ys[skip:])
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad

    pw = _CHART_W - _CHART_PAD_L - _CHART_PAD_R
    ph = _CHART_H - _CHART_PAD_T - _CHART_PAD_B

    def sx(x: float) -> float:
        return _CHART_PAD_L + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return _CHART_PAD_T + (1 - (y - y0) / (y1 - y0)) * ph

    def polyline(values: list[float], extra: str) -> str:
        step = max(1, len(values) // _CHART_MAX_POINTS)
        idx = list(range(0, len(values), step))
        if idx[-1] != len(values) - 1:
            idx.append(len(values) - 1)
        coords = " ".join(f"{sx(xs[i]):.1f},{sy(values[i]):.1f}" for i in idx)
        return (
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-linejoin="round" stroke-linecap="round" {extra}/>'
        )

    clip = f"rvc-clip-{uid}"
    parts = [
        f'<svg viewBox="0 0 {_CHART_W} {_CHART_H}" class="rvc-chart-svg" '
        f'role="img" aria-label="{html.escape(name)}">',
        f'<defs><clipPath id="{clip}"><rect x="{_CHART_PAD_L}" y="{_CHART_PAD_T}" '
        f'width="{pw}" height="{ph}"/></clipPath></defs>',
    ]
    ystep = _nice_step(y1 - y0)
    yv = math.ceil(y0 / ystep) * ystep
    while yv <= y1:
        y = sy(yv)
        parts.append(
            f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
            f'y1="{y:.1f}" y2="{y:.1f}" class="rvc-chart-grid"/>'
            f'<text x="{_CHART_PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="rvc-chart-tick">{_nice_num(yv)}</text>'
        )
        yv += ystep
    xstep = _nice_step(x1 - x0, 6)
    xv = math.ceil(x0 / xstep) * xstep
    while xv <= x1 + 1e-9:
        parts.append(
            f'<text x="{sx(xv):.1f}" y="{_CHART_H - 8}" text-anchor="middle" '
            f'class="rvc-chart-tick">{_nice_num(xv)}</text>'
        )
        xv += xstep
    parts.append(
        f'<line x1="{_CHART_PAD_L}" x2="{_CHART_W - _CHART_PAD_R}" '
        f'y1="{_CHART_PAD_T + ph}" y2="{_CHART_PAD_T + ph}" class="rvc-chart-axis"/>'
        f'<g clip-path="url(#{clip})">'
        + polyline(ys, 'stroke-width="1.5" stroke-opacity="0.35"')
        + polyline(smooth, 'stroke-width="3"')
        + "</g></svg>"
    )

    first, last = smooth[0], smooth[-1]
    change = f"{(last - first) / first * 100:+.0f}%" if first else ""
    head = (
        f'<div class="rvc-chart-head"><span><i style="background:{color}"></i>'
        f"<b>{html.escape(name)}</b></span>"
        f"<span>teraz <b>{_nice_num(ys[-1])}</b>"
        + (f" · od startu <b>{change}</b>" if change else "")
        + "</span></div>"
    )
    return f'<div class="rvc-chart-panel">{head}{"".join(parts)}</div>'


def loss_chart_svg(rows: list[dict]) -> str:
    """Wykres strat jako statyczny SVG (bez JS), osobny panel na każdą serię.

    Zamiast ``gr.LinePlot``: wykres Vega w Gradio 5 przebudowuje się przy
    zmianach rozmiaru i danych - przy pierwszych punktach strat w trakcie
    streamingu treningu zawieszał całą kartę przeglądarki.
    """
    series: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        series.setdefault(r["seria"], []).append((float(r["epoka"]), float(r["strata"])))
    series = {k: v for k, v in series.items() if len(v) >= 2}
    if not series:
        return (
            '<div class="rvc-chart-empty">Wykres strat pojawi się po pierwszych '
            "50 krokach treningu.</div>"
        )
    panels = "".join(
        _loss_panel(name, pts, _CHART_COLORS[i % len(_CHART_COLORS)], i)
        for i, (name, pts) in enumerate(series.items())
    )
    return (
        f'<div class="rvc-chart">{panels}'
        '<div class="rvc-chart-legend">Oś pozioma: epoka. Gruba linia: średnia '
        "krocząca, cienka: surowe wartości.</div></div>"
    )


def _card(pct: float, title: str, details: list[str]) -> str:
    items = "".join(f"<li>{d}</li>" for d in details)
    return (
        '<div class="rvc-progress">'
        f'<div class="rvc-progress-title"><span>{title}</span>'
        f'<span class="rvc-progress-pct">{pct * 100:.0f}%</span></div>'
        f'<div class="rvc-progress-bar"><div style="width:{pct * 100:.1f}%"></div></div>'
        f'<ul class="rvc-progress-details">{items}</ul>'
        "</div>"
    )


def _children_map() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8") as fh:
                stat = fh.read()
        except OSError:
            continue
        # Nazwa procesu w nawiasach może zawierać spacje - PPID jest za ostatnim ')'.
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    return children


def _kill_tree(pid: int, grace_s: float = 10.0) -> None:
    """Zabija proces z potomkami (core.py → train.py → procesy multiprocessing).

    Bez osobnej grupy procesów (``start_new_session``), żeby przerwanie
    komórki w Colabie nadal zabijało cały trening razem z UI.
    """
    if not os.path.isdir("/proc"):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        return
    tree = [pid]
    children = _children_map()
    i = 0
    while i < len(tree):
        tree.extend(children.get(tree[i], []))
        i += 1
    for p in reversed(tree):
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not any(os.path.exists(f"/proc/{p}") and not _is_zombie(p) for p in tree):
            return
        time.sleep(0.5)
    for p in tree:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


_ACTIVE: dict[str, TrainingMonitor] = {}
_ACTIVE_LOCK = threading.Lock()


def register(monitor: TrainingMonitor) -> None:
    with _ACTIVE_LOCK:
        if monitor.model_name in _ACTIVE:
            raise RuntimeError(f"Trening modelu „{monitor.model_name}” już trwa.")
        _ACTIVE[monitor.model_name] = monitor


def unregister(monitor: TrainingMonitor) -> None:
    with _ACTIVE_LOCK:
        if _ACTIVE.get(monitor.model_name) is monitor:
            del _ACTIVE[monitor.model_name]


def get_active(model_name: str) -> TrainingMonitor | None:
    with _ACTIVE_LOCK:
        return _ACTIVE.get(model_name)


def active_names() -> list[str]:
    with _ACTIVE_LOCK:
        return list(_ACTIVE)
