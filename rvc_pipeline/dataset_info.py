"""Podsumowanie próbek treningowych przed startem (Zakładka 2).

Szacunek czasu oparty na **własnym pomiarze** w Colabie (T4, dziennik
Kroki 31-32): ~13 min audio, batch 12 → 41 kroków/epokę, tempo stabilne
~71 s/epokę, a pierwsze ~25 epok średnio o ~70% wolniejsze (rozgrzewka) -
100 epok zajęło ~2 h 19 min. Czas epoki skaluje się z długością datasetu
(liczba próbek), a nie z batch size (ten sam materiał na GPU).
"""

from __future__ import annotations

import math
import shutil
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from rvc_pipeline.applio_train import ApplioTrainingError, _prepare_dataset_dir

_REF_MINUTES = 13.0
_REF_STEPS_BATCH12 = 41
_REF_EPOCH_S = 71.0
_WARMUP_EPOCHS = 25
_WARMUP_EXTRA = 0.7
_LONG_TRAINING_S = 3 * 3600


def _duration(path: Path) -> float | None:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


@lru_cache(maxsize=16)
def _measure(key: tuple[tuple[str, int, float], ...]) -> dict:
    paths = [k[0] for k in key]
    work = Path(tempfile.mkdtemp(prefix="rvc_dsinfo_"))
    try:
        dataset = _prepare_dataset_dir(paths, work)
        durations, unreadable = [], 0
        for f in sorted(dataset.iterdir()):
            d = _duration(f)
            if d is None:
                unreadable += 1
            else:
                durations.append(d)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"durations": durations, "unreadable": unreadable}


def _fmt_s(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {s:02d} s"


def _source_counts(paths: list[str]) -> str:
    zips = sum(1 for p in paths if Path(p).suffix.lower() == ".zip")
    audio = len(paths) - zips
    parts = []
    if zips:
        parts.append(f"{zips} ZIP" + ("" if zips == 1 else "-y" if zips < 5 else "-ów"))
    if audio:
        parts.append(f"{audio} plik(i) audio")
    return " + ".join(parts) or "brak"


def estimate_training_seconds(total_minutes: float, epochs: int) -> tuple[float, float]:
    """``(tempo_stabilne_s_na_epokę, łączny_czas_s)`` - szacunek dla T4."""
    pace = _REF_EPOCH_S * total_minutes / _REF_MINUTES
    total = pace * (epochs + _WARMUP_EXTRA * min(epochs, _WARMUP_EPOCHS))
    return pace, total


def summarize_samples(
    sample_paths: list[str] | None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> str:
    """Markdown z podsumowaniem próbek i szacunkiem czasu (pusty, gdy brak plików)."""
    paths = [p for p in (sample_paths or []) if p]
    if not paths:
        return ""
    try:
        key = tuple(
            (str(Path(p)), Path(p).stat().st_size, Path(p).stat().st_mtime)
            for p in paths
        )
        info = _measure(key)
    except (ApplioTrainingError, OSError, zipfile.BadZipFile) as exc:
        return f"**Próbki:** nie udało się odczytać ({exc})."

    durations = info["durations"]
    if not durations:
        return "**Próbki:** brak czytelnych plików audio."
    total = sum(durations)
    minutes = total / 60
    lines = [
        f"**Próbki:** {_source_counts(paths)} → **{len(durations)} plików**, "
        f"łącznie **{_fmt_s(total)}** (średnio {total / len(durations):.1f} s; "
        f"od {min(durations):.1f} s do {max(durations):.1f} s)."
    ]
    if info["unreadable"]:
        lines.append(f"⚠ {info['unreadable']} plik(ów) nie dało się odczytać - zostaną pominięte przez Applio albo spowodują błąd.")
    if minutes < 5:
        lines.append("⚠ Bardzo mało materiału - model będzie słabo oddawał barwę (zalecane 10-30 min czystej mowy).")
    elif minutes < 10:
        lines.append("⚠ Mało materiału - zalecane 10-30 min czystej mowy.")
    elif minutes > 60:
        lines.append("⚠ Dużo materiału - każda epoka będzie długa; 20-40 min zwykle wystarcza.")

    if epochs and batch_size:
        pace, est = estimate_training_seconds(minutes, int(epochs))
        steps = math.ceil(_REF_STEPS_BATCH12 * (minutes / _REF_MINUTES) * 12 / int(batch_size))
        lines.append(
            f"**Szacunek (Colab T4, {int(epochs)} epok, batch {int(batch_size)}):** "
            f"~{steps} kroków/epokę, ~{_fmt_s(pace)}/epokę po rozgrzewce, "
            f"łącznie ok. **{_fmt_s(est)}** (+ kilka minut przygotowania danych)."
        )
        if est > _LONG_TRAINING_S:
            lines.append(
                "⚠ Może nie zmieścić się w jednej sesji darmowego Colaba - włącz "
                "auto-kopię G/D na Drive i dokończ trening w kolejnej sesji (Wznów)."
            )
    lines.append("<sub>Szacunek z własnego pomiaru (13 min audio → ~1:11/epokę na T4); rzeczywisty czas zależy od przydzielonego GPU.</sub>")
    return "\n\n".join(lines)
