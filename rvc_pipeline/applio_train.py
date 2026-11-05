"""Cienki wrapper na trening Applio (RVC) - Zakładka 2 (trening modelu).

Ten sam mechanizm izolacji co przy inferencji (`applio_engine.py`): Applio
**nie jest importowane** do procesu naszego UI - wołane jest jako osobne
procesy (subprocess) w osobnym venv (`APPLIO_DIR` / `APPLIO_PYTHON`).

Applio dzieli trening na 3 kroki CLI, wołane po kolei:
    core.py preprocess  - segmentacja / przygotowanie datasetu
    core.py extract      - ekstrakcja cech (F0 + embeddingi mówcy)
    core.py train        - trening GAN (Applio sam woła `index` na końcu)

Wynik trafia do `logs/<model_name>/` w klonie Applio: checkpointy treningowe
(`G_*`/`D_*`) oraz **model inferencyjny** (`{name}_*e_*s.pth` z kluczem
`config`) + plik `.index`. Do ZIP pakujemy wyłącznie model inferencyjny
(+ index) - `G_*.pth` nie nadaje się do `core.py infer` (KeyError: config).

Wznawianie treningu: Applio (`rvc/train/train.py`, blok „Load checkpoint if
available”) samo ładuje najnowsze `G_*.pth` + `D_*.pth` z `logs/<model_name>/`
i kontynuuje od zapisanej epoki + 1 aż do `--total-epoch`. Trenujemy z
`--save-only-latest`, więc w katalogu jest zawsze jedna para
`G_2333333.pth` / `D_2333333.pth` (nadpisywana co `--save-every-epoch` epok).
W nowej sesji Colaba (`/content` wyczyszczone) wystarczy ponownie
przetworzyć te same próbki i podłożyć zapisane wcześniej G/D.

Źródła:
  - Applio CLI (`core.py preprocess/extract/train`, argumenty):
    https://docs.applio.org/reference/cli/
  - Repo / tag: https://github.com/IAHispano/Applio (LICENSE: MIT)
  - Kolejność preprocess→extract→train→index: `core.py` (funkcje
    `run_preprocess_script`/`run_extract_script`/`run_train_script`,
    tag 3.6.5) - `run_train_script` woła `run_index_script` po treningu.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

from rvc_pipeline.checkpoints import (
    CheckpointPair,
    checkpoint_drive_name,
    copy_atomic,
    drive_checkpoint_dir,
    drive_dest_dir,
    drive_mounted,
    latest_training_checkpoint,
    list_checkpoint_pairs,
    wait_for_checkpoint_written,
)
from rvc_pipeline import save_log
from rvc_pipeline.train_monitor import (
    TrainingMonitor,
    TrainingStopped,
    _kill_tree,
    get_active,
)
from rvc_pipeline.applio_engine import (
    ApplioConfig,
    ApplioConfigError,
    DEFAULT_EMBEDDER_MODEL,
    EMBEDDER_MODELS,
    F0_METHODS,
    probe_torch_device,
)


class ApplioTrainingError(RuntimeError):
    """Jeden z kroków treningu Applio (preprocess/extract/train) się nie powiódł."""


# Te same wartości co dropdown SR w Zakładce 1 (audio_tools.TARGET_SAMPLE_RATES)
# i jedyne dozwolone przez `core.py preprocess`/`train` (Click Choice).
SAMPLE_RATES: tuple[int, ...] = (32000, 40000, 48000)
DEFAULT_SAMPLE_RATE = 40000

# `core.py train --save-every-epoch` przyjmuje 1-100 (Click IntRange).
SAVE_EVERY_MIN = 1
SAVE_EVERY_MAX = 100
DEFAULT_SAVE_EVERY = 25

# Sufiks plików przy `--save-only-latest` (stała w Applio `train.py`).
_LATEST_CKPT_SUFFIX = "2333333"

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
# "ł"/"Ł" nie mają dekompozycji NFKD (to nie znak z akcentem, a przekreślone
# l) - bez tej mapy unicodedata.normalize je po prostu usuwa.
_POLISH_STROKE_MAP = str.maketrans({"ł": "l", "Ł": "L"})


def sanitize_model_name(name: str) -> str:
    """Nazwa modelu = nazwa katalogu w `logs/` Applio - tylko bezpieczne znaki.

    Polskie znaki (ą, ę, ł, ó…) są najpierw sprowadzane do ASCII (ą→a, ł→l…),
    żeby „Mój Głos” dało sensowne ``Moj_Glos``, nie ``M_j_G_os``.
    """
    pre = (name or "").strip().translate(_POLISH_STROKE_MAP)
    ascii_name = unicodedata.normalize("NFKD", pre)
    ascii_name = ascii_name.encode("ascii", "ignore").decode("ascii")
    cleaned = _NAME_RE.sub("_", ascii_name).strip("_-")
    if not cleaned:
        raise ApplioTrainingError(
            "Podaj nazwę modelu: litery, cyfry, myślnik, podkreślenie."
        )
    return cleaned


def _unique_target(dataset_dir: Path, name: str, tag: str) -> Path:
    target = dataset_dir / name
    if not target.exists():
        return target
    safe_tag = _NAME_RE.sub("_", tag).strip("_") or "plik"
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 1
    target = dataset_dir / f"{safe_tag}_{stem}{suffix}"
    while target.exists():
        counter += 1
        target = dataset_dir / f"{safe_tag}_{stem}_{counter}{suffix}"
    return target


def _prepare_dataset_dir(sample_paths: list[str], work_dir: Path) -> Path:
    """Kopiuje wgrane próbki do jednego katalogu (dataset pod `core.py preprocess`).

    Obsługuje pliki audio podane pojedynczo oraz ZIP-y (np. „Pobierz wszystko
    (ZIP)” z Zakładki 1) - z ZIP-a wyciągane są tylko pliki audio. Kilka ZIP-ów
    z Zakładki 1 ma te same nazwy próbek (``sample_0001.wav``…), więc przy
    kolizji nazwa dostaje prefiks z nazwy archiwum - inaczej próbki z drugiego
    ZIP-a po cichu nadpisałyby pierwsze.
    """
    dataset_dir = work_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    found_any = False

    for raw_path in sample_paths:
        p = Path(raw_path)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = Path(info.filename).name
                    if Path(name).suffix.lower() not in _AUDIO_EXTS:
                        continue
                    target = _unique_target(dataset_dir, name, p.stem)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found_any = True
        elif p.suffix.lower() in _AUDIO_EXTS:
            shutil.copy2(p, _unique_target(dataset_dir, p.name, p.parent.name))
            found_any = True

    if not found_any:
        raise ApplioTrainingError(
            "Nie znaleziono żadnych plików audio wśród wgranych próbek. "
            "Obsługiwane: WAV, MP3, FLAC, OGG, M4A albo ZIP z takimi plikami."
        )

    return dataset_dir


_TQDM_BAR_RE = re.compile(r"\d+%\|")


# Odświeżenie UI najwyżej co _YIELD_INTERVAL_S (gdy przyszły nowe linie) albo
# co _HEARTBEAT_S przy ciszy. Yield na każdą linię tqdm (-u → kilka/s) zalewał
# przeglądarkę aktualizacjami przez link gradio.live.
_YIELD_INTERVAL_S = 1.0
_HEARTBEAT_S = 5.0
_LOG_TAIL_LINES = 300
_LOG_KEEP_LINES = 5000
_DRAIN_MAX_LINES = 5000
_SILENCE_HINT_S = 20.0
_SILENCE_NOTE = (
    "… Applio nic nie wypisuje - bywa to zapis G/D albo ewaluacja, ale przy tym samym "
    "kroku przez wiele minut to raczej zawieszenie. Zatrzymaj trening albo "
    "Interrupt w Colabie, potem wznów z checkpointu, jeśli jest."
)


def _unbuffered_cmd(cmd: list[str]) -> list[str]:
    """``python -u …`` - bez tego Applio buforuje stdout i UI „zamiera” na minuty."""
    if len(cmd) >= 2 and not cmd[1].startswith("-"):
        return [cmd[0], "-u", *cmd[1:]]
    return list(cmd)


def _start_line_reader(stream) -> queue.Queue:
    """Czyta stdout procesu w osobnym wątku; ``None`` w kolejce = koniec strumienia.

    Nie ``select()`` + ``readline()``: ``readline`` wczytuje do bufora
    ``TextIOWrapper`` cały dostępny blok, a ``select`` widzi tylko deskryptor
    - linie z bufora czekały na kolejne dane i log w UI coraz bardziej
    odstawał od faktycznego treningu (np. „epoka 2, 39/41”, gdy Applio było
    już kilka epok dalej).
    """
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        try:
            for raw in stream:
                lines.put(raw)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    threading.Thread(target=pump, name="applio-stdout", daemon=True).start()
    return lines


def _stream_step(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    step_name: str,
    monitor: TrainingMonitor | None = None,
) -> Iterator[str]:
    """Uruchamia krok Applio i emituje narastający log na żywo (ostatnie linie).

    Trening bywa długi (dziesiątki minut / godziny) - ``Popen`` + wątek
    czytający stdout pozwala pokazać w UI, że coś się dzieje. Kolejne
    odświeżenia paska tqdm (``\\r`` → osobne „linie”) zastępują poprzednie,
    żeby nie wypchnęły z okna właściwego logu. Każdy yield to cały log od
    nowa, ale najwyżej raz na ``_YIELD_INTERVAL_S``.
    """
    if monitor is not None and monitor.stop_requested:
        raise TrainingStopped()

    run_cmd = _unbuffered_cmd(cmd)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Wątki OpenMP na proces (Applio + 4 workery DataLoadera); Colab ma ~2 vCPU.
    env.setdefault("OMP_NUM_THREADS", "2")

    try:
        proc = subprocess.Popen(
            run_cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        raise ApplioConfigError(
            f"Nie udało się uruchomić {cmd[0]} - sprawdź APPLIO_PYTHON. Błąd: {exc}"
        ) from exc
    if monitor is not None:
        monitor.attach(proc)

    collected: list[str] = [f"Polecenie: {' '.join(cmd)}"]

    def tail() -> str:
        return "\n".join(collected[-_LOG_TAIL_LINES:])

    try:
        assert proc.stdout is not None
        lines = _start_line_reader(proc.stdout)
        start = time.monotonic()
        last_line_at = start
        last_yield_at = start
        silence_noted = False
        changed = True
        eof = False
        stop_wait_noted = False

        while not eof:
            if (
                monitor is not None
                and monitor.stop_requested
                and monitor.checkpoint_save_pending
                and proc.poll() is None
            ):
                if not stop_wait_noted:
                    stop_wait_noted = True
                    changed = True
                    monitor.phase = "Zatrzymywanie: czekam na dokończenie zapisu G/D…"
                    collected.append(
                        "… Zatrzymanie: Applio kończy zapis checkpointu G/D, "
                        "potem trening zostanie przerwany."
                    )
            elif monitor is not None and monitor.stop_requested:
                if proc.poll() is None:
                    _kill_tree(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise TrainingStopped()

            if time.monotonic() - start > timeout_s:
                _kill_tree(proc.pid)
                raise ApplioTrainingError(
                    f"Krok '{step_name}' przekroczył limit czasu {timeout_s} s."
                )

            try:
                raw = lines.get(timeout=_YIELD_INTERVAL_S / 2)
            except queue.Empty:
                raw = ""
            drained = 0
            while raw is not None and drained < _DRAIN_MAX_LINES:
                line = raw.rstrip()
                if line:
                    last_line_at = time.monotonic()
                    silence_noted = False
                    changed = True
                    if _TQDM_BAR_RE.search(line) and _TQDM_BAR_RE.search(collected[-1]):
                        collected[-1] = line
                    else:
                        collected.append(line)
                    if monitor is not None:
                        monitor.feed(line)
                drained += 1
                try:
                    raw = lines.get_nowait()
                except queue.Empty:
                    break
            if raw is None:
                eof = True
            if len(collected) > _LOG_KEEP_LINES:
                del collected[1:-_LOG_KEEP_LINES]

            now = time.monotonic()
            if not silence_noted and now - last_line_at >= _SILENCE_HINT_S:
                silence_noted = True
                changed = True
                if monitor is not None and monitor.training_started:
                    monitor.phase = (
                        "Trwa zapis checkpointu albo ewaluacja…"
                    )
                if collected[-1] != _SILENCE_NOTE:
                    collected.append(_SILENCE_NOTE)

            since = now - last_yield_at
            if eof or (changed and since >= _YIELD_INTERVAL_S) or since >= _HEARTBEAT_S:
                changed = False
                last_yield_at = now
                yield tail()

        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            proc.wait()
    finally:
        if monitor is not None:
            monitor.detach()
        if proc.poll() is None:
            # Zamknięty generator (rozłączona karta, anulowanie w Gradio) nie
            # może zostawić treningu w tle bez UI - ale nie w środku zapisu G/D.
            deadline = time.monotonic() + 240
            while (
                monitor is not None
                and monitor.checkpoint_save_pending
                and time.monotonic() < deadline
            ):
                time.sleep(1)
            _kill_tree(proc.pid)

    if monitor is not None and monitor.stop_requested and proc.returncode != 0:
        raise TrainingStopped()
    full_log = "\n".join(collected)
    if proc.returncode != 0:
        hint = ""
        if proc.returncode < 0:
            hint = (
                f"\nProces zabity przez system, sygnał {-proc.returncode} - "
                "częste przy braku RAM na darmowym Colabie."
            )
        raise ApplioTrainingError(
            f"Krok '{step_name}' zakończył się błędem, "
            f"kod {proc.returncode}.{hint}\n\n{full_log}"
        )


def _ensure_applio_assets_config(applio_dir: Path) -> None:
    """Zapewnia istnienie ``assets/config.json`` w klonie Applio.

    **To jest prawdziwa przyczyna**, dla której trening produkował tylko
    ``G_*.pth`` bez małego modelu inferencyjnego: Applio ``core.py train``
    **sam** woła ``extract_model()`` na końcowej epoce (niezależnie od flagi
    ``--save-every-weights`` - patrz ``rvc/train/train.py``, blok „Final
    model”), ale ``extract_model()`` czyta ten plik tylko po
    ``model_author`` i przy jego braku łapie wyjątek w gołym
    ``except Exception`` i **tylko printuje** błąd - bez zapisania pliku,
    bez podniesienia wyjątku, bez informacji w naszym logu treningu.
    W tym środowisku Colab pliku nie było (`core.py prerequisites` go nie
    tworzy), więc ekstrakcja padała po każdym treningu.
    """
    cfg = applio_dir / "assets" / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.is_file():
        cfg.write_text('{"model_author": null}\n', encoding="utf-8")


_APPLIO_TRAIN_PATCH_MARK = "RVC_COLAB_PL_PATCH"


# Ustawienia DataLoadera zmieniane przez wcześniejsze wersje patcha (V1: 2
# workery, V2: 0 workerów) - przywracane do oryginału Applio 3.6.5. Zawieszenia
# „w środku epoki” okazały się zamrożeniem przeglądarki i opóźnionym logiem
# (patrz _start_line_reader), a 0 workerów tylko spowalniało trening.
_DATALOADER_RESTORE = (
    (re.compile(r"num_workers=\d+,[ \t]*# RVC_COLAB_PL_PATCH[^\n]*"), "num_workers=4,"),
    (
        re.compile(r"persistent_workers=False,[ \t]*# RVC_COLAB_PL_PATCH[^\n]*"),
        "persistent_workers=True,",
    ),
    (re.compile(r"# prefetch_factor=8,[ \t]*# RVC_COLAB_PL_PATCH[^\n]*"), "prefetch_factor=8,"),
    (
        re.compile(r"from torch\.utils\.data import DataLoader[ \t]*# RVC_COLAB_PL_PATCH[^\n]*"),
        "from torch.utils.data import DataLoader",
    ),
)


def _patch_applio_train_for_colab(applio_dir: Path) -> None:
    """Dostosowanie ``rvc/train/train.py`` pod Colab (idempotentne).

    Wyłącza mid-run ``save_every_weights`` - eksport małego modelu przy
    każdym ``save_every_epoch`` potrafi zająć wiele minut i wygląda jak
    zawieszenie UI. Model inferencyjny i tak powstaje na końcu (blok
    „Final model” w Applio) + nasz fallback ``extract_model``. Cofa też
    zmiany DataLoadera z poprzednich wersji patcha (``_DATALOADER_RESTORE``).
    """
    path = applio_dir / "rvc" / "train" / "train.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    original = text

    if "save_every_weights = False  # bez mid-run extract" not in text:
        text = text.replace(
            "save_every_weights = _strtobool(sys.argv[10])",
            "_strtobool(sys.argv[10])  # CLI ignorowane - "
            + _APPLIO_TRAIN_PATCH_MARK
            + "\nsave_every_weights = False  # bez mid-run extract; final model zostaje",
            1,
        )

    for pattern, replacement in _DATALOADER_RESTORE:
        text = pattern.sub(replacement, text)

    if text == original:
        return
    path.write_text(text, encoding="utf-8")


def _is_training_checkpoint_name(name: str) -> bool:
    """True dla checkpointów treningowych G_*/D_* (nie nadają się do inferencji)."""
    n = name.lower()
    return n.startswith("g_") or n.startswith("d_")


def _list_infer_pths(model_dir: Path) -> list[Path]:
    """Pliki .pth eksportowane pod konwersję (bez G_/D_)."""
    return sorted(
        p
        for p in model_dir.rglob("*.pth")
        if p.is_file() and not _is_training_checkpoint_name(p.name)
    )


def _list_generator_ckpts(model_dir: Path) -> list[Path]:
    """Checkpointy generatora G_*.pth (do ekstrakcji modelu inferencyjnego)."""
    return sorted(
        p
        for p in model_dir.rglob("*.pth")
        if p.is_file() and p.name.lower().startswith("g_")
    )


def _extract_infer_pth_from_generator(
    config: ApplioConfig,
    model_dir: Path,
    model_name: str,
    g_path: Path,
) -> Path:
    """Buduje mały model inferencyjny z G_*.pth (Applio ``extract_model``).

    ``G_*.pth`` to checkpoint treningowy (bez klucza ``config``) - Applio
    ``core.py infer`` na nim pada z ``KeyError: 'config'``. Ekstrakcja
    dokłada metadane (config, sr, f0…) i zapisuje plik ``{name}_*e_*s.pth``.
    """
    out_path = model_dir / f"{model_name}_extracted_from_{g_path.stem}.pth"
    # Skrypt wołany w venv Applio (nasz UI nie ma torcha / kodu Applio).
    helper = r"""
import json, re, sys
from pathlib import Path
import torch
from rvc.train.process.extract_model import extract_model
from rvc.train.utils import HParams

model_dir = Path(sys.argv[1])
model_name = sys.argv[2]
g_path = Path(sys.argv[3])
out_path = Path(sys.argv[4])
applio_dir = Path(sys.argv[5])

# Zabezpieczenie (main procesu też to robi przed `core.py train`, ale ta
# funkcja bywa wołana samodzielnie dla starszych/uszkodzonych logs/ z czasu
# przed tym fixem) - bez tego extract_model pada po cichu (KeyError „config”
# nigdy nie powstanie, bo plik .pth po prostu nie zostanie zapisany).
assets_cfg = applio_dir / "assets" / "config.json"
assets_cfg.parent.mkdir(parents=True, exist_ok=True)
if not assets_cfg.is_file():
    assets_cfg.write_text('{"model_author": null}\n', encoding="utf-8")

cfg_path = model_dir / "config.json"
if not cfg_path.is_file():
    raise SystemExit(
        f"Brak {cfg_path} - nie da się zbudować modelu z G_ bez config.json po preprocess."
    )

with open(cfg_path, encoding="utf-8") as f:
    hps = HParams(**json.load(f))

ckpt = torch.load(g_path, map_location="cpu", weights_only=True)
if not isinstance(ckpt, dict) or "model" not in ckpt:
    raise SystemExit(f"{g_path.name} nie wygląda na checkpoint G_ (brak klucza 'model').")

state = ckpt["model"]
epoch = int(ckpt.get("iteration") or 0)
step = 0
m = re.match(r"G_(\d+)\.pth$", g_path.name, flags=re.IGNORECASE)
if m:
    step = int(m.group(1))
elif epoch:
    step = epoch

extract_model(
    ckpt=state,
    sr=hps.data.sample_rate,
    name=model_name,
    model_path=str(out_path),
    epoch=epoch,
    step=step,
    hps=hps,
    vocoder="HiFi-GAN",
)
if not out_path.is_file():
    raise SystemExit(
        "extract_model nie zapisał pliku wynikowego "
        "(często: brak assets/config.json albo błąd w logu powyżej)."
    )
print(out_path)
"""
    try:
        proc = subprocess.run(
            [
                config.python_bin,
                "-c",
                helper,
                str(model_dir),
                model_name,
                str(g_path),
                str(out_path),
                str(config.applio_dir),
            ],
            cwd=str(config.applio_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except OSError as exc:
        raise ApplioConfigError(
            f"Nie udało się uruchomić ekstrakcji modelu (APPLIO_PYTHON): {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ApplioTrainingError(
            "Timeout przy ekstrakcji modelu inferencyjnego z G_*.pth."
        ) from exc

    if proc.returncode != 0 or not out_path.is_file():
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ApplioTrainingError(
            "Nie udało się zbudować modelu inferencyjnego z checkpointu "
            f"{g_path.name}.\n{detail}"
        )
    return out_path


def _pick_final_model(
    config: ApplioConfig, model_dir: Path, model_name: str
) -> tuple[Path, Path | None]:
    """Wybiera finalny plik `.pth` **do inferencji** i opcjonalny `.index`.

    Nigdy nie pakuje ``G_*.pth`` / ``D_*.pth`` - to checkpointy treningowe.
    Jeśli Applio nie zapisało małego modelu, budujemy go z najnowszego ``G_*``.
    """
    if not model_dir.is_dir():
        raise ApplioTrainingError(
            f"Trening zakończony, ale katalog {model_dir} nie istnieje - "
            "sprawdź log wyżej."
        )

    infer_pths = _list_infer_pths(model_dir)
    if not infer_pths:
        generators = _list_generator_ckpts(model_dir)
        if not generators:
            raise ApplioTrainingError(
                f"Trening zakończony, ale w {model_dir} nie ma ani modelu "
                "inferencyjnego (*.pth bez G_/D_), ani checkpointu G_*.pth. "
                "Sprawdź log treningu wyżej."
            )
        latest_g = max(generators, key=lambda p: p.stat().st_mtime)
        infer_pths = [
            _extract_infer_pth_from_generator(config, model_dir, model_name, latest_g)
        ]

    chosen = max(infer_pths, key=lambda p: p.stat().st_mtime)

    index_candidates = sorted(
        model_dir.rglob("*.index"), key=lambda p: p.stat().st_mtime
    )
    index = index_candidates[-1] if index_candidates else None
    return chosen, index


def _has_extracted_dataset(model_dir: Path) -> bool:
    """True, gdy `preprocess` + `extract` już przeszły dla tego katalogu."""
    filelist = model_dir / "filelist.txt"
    return (
        (model_dir / "config.json").is_file()
        and filelist.is_file()
        and filelist.stat().st_size > 0
    )


def _model_dir_sample_rate(model_dir: Path) -> int | None:
    try:
        with open(model_dir / "config.json", encoding="utf-8") as f:
            return int(json.load(f)["data"]["sample_rate"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def read_checkpoint_epoch(config: ApplioConfig, ckpt_path: Path) -> int | None:
    """Epoka zapisana w checkpoincie G_/D_ (klucz ``iteration`` w Applio
    ``save_checkpoint``). Czytane w venv Applio - nasz UI nie ma torcha."""
    helper = (
        "import sys, torch\n"
        "c = torch.load(sys.argv[1], map_location='cpu', weights_only=True)\n"
        "if not isinstance(c, dict) or 'model' not in c:\n"
        "    raise SystemExit('brak klucza model')\n"
        "print(int(c.get('iteration') or 0))\n"
    )
    try:
        proc = subprocess.run(
            [config.python_bin, "-c", helper, str(ckpt_path)],
            cwd=str(config.applio_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def _install_uploaded_checkpoints(
    model_dir: Path, g_upload: str, d_upload: str
) -> None:
    """Podkłada wgrane G/D pod nazwy, które Applio wczyta przy starcie treningu."""
    for prefix in ("G", "D"):
        for old in model_dir.glob(f"{prefix}_*.pth"):
            old.unlink()
    shutil.copy2(g_upload, model_dir / f"G_{_LATEST_CKPT_SUFFIX}.pth")
    shutil.copy2(d_upload, model_dir / f"D_{_LATEST_CKPT_SUFFIX}.pth")


def run_training(
    model_name: str,
    sample_paths: list[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    epochs: int = 200,
    batch_size: int = 8,
    f0_method: str = "rmvpe",
    embedder_model: str = DEFAULT_EMBEDDER_MODEL,
    save_every: int = DEFAULT_SAVE_EVERY,
    resume: bool = False,
    resume_g_path: str | None = None,
    resume_d_path: str | None = None,
    auto_backup: bool = False,
    monitor: TrainingMonitor | None = None,
) -> Iterator[tuple[str, str, str | None]]:
    """Generator: ``(etap, log_narastający, zip_gotowego_modelu_lub_None)``.

    Etapy: ``start`` → ``preprocess`` → ``extract`` → ``train`` → ``done``.
    Rzuca ``ApplioConfigError`` / ``ApplioTrainingError`` - łapane w
    ``rvc_pipeline.pipeline.run_train_ui`` i tłumaczone na komunikat PL w UI.

    ``epochs`` to **łączna** docelowa liczba epok - przy wznowieniu od epoki
    100 i ``epochs=200`` Applio dotrenowuje epoki 101-200.

    Wznowienie (``resume=True``):
      * ta sama sesja - w ``logs/<nazwa>/`` są już dane i G/D; próbki
        nie są potrzebne (preprocess/extract są pomijane),
      * nowa sesja - wgrać **te same** próbki + pliki G i D (np. z Drive);
        próbki są przetwarzane od nowa, a G/D podkładane przed treningiem.

    ``monitor`` (opcjonalny) dostaje każdą linię logu - postęp/ETA/straty,
    przycisk „Zatrzymaj trening” i auto-kopię G/D na Drive (``auto_backup``).
    Po zatrzymaniu pakowany jest model z ostatniego checkpointu (etap
    ``stopped``).
    """
    config = ApplioConfig.from_env()
    name = sanitize_model_name(model_name)
    # Musi być PRZED `core.py train` - inaczej Applio ubija ekstrakcję
    # finalnego modelu po cichu (patrz docstring _ensure_applio_assets_config).
    _ensure_applio_assets_config(config.applio_dir)
    _patch_applio_train_for_colab(config.applio_dir)

    if sample_rate not in SAMPLE_RATES:
        raise ApplioTrainingError(
            f"Nieobsługiwany sample rate: {sample_rate}. Dozwolone: {SAMPLE_RATES}."
        )
    method = (f0_method or "rmvpe").strip().lower()
    if method not in F0_METHODS:
        raise ApplioTrainingError(
            f"Nieznana metoda F0: '{f0_method}'. Dozwolone: {', '.join(F0_METHODS)}."
        )
    if embedder_model not in EMBEDDER_MODELS:
        raise ApplioTrainingError(
            f"Nieznany model embeddingu: '{embedder_model}'. "
            f"Dozwolone: {', '.join(EMBEDDER_MODELS)}."
        )
    save_every = int(save_every or DEFAULT_SAVE_EVERY)
    if not SAVE_EVERY_MIN <= save_every <= SAVE_EVERY_MAX:
        raise ApplioTrainingError(
            f"Zapis checkpointu co {save_every} epok - dozwolone "
            f"{SAVE_EVERY_MIN}-{SAVE_EVERY_MAX}."
        )
    epochs = int(epochs)

    model_dir = config.applio_dir / "logs" / name
    has_upload = bool(resume_g_path or resume_d_path)
    if has_upload and not (resume_g_path and resume_d_path):
        raise ApplioTrainingError(
            "Do wznowienia z plików wgraj oba checkpointy, G i D - "
            "Applio potrzebuje pary z tej samej epoki."
        )

    existing_g = latest_training_checkpoint(model_dir, "G")
    if not resume and existing_g is not None:
        raise ApplioTrainingError(
            f"Model „{name}” ma już checkpoint w tej sesji: {existing_g.name}. "
            "Zaznacz „Wznów trening”, żeby go kontynuować, albo podaj inną "
            "nazwę dla nowego modelu."
        )
    if not resume and model_dir.exists():
        # Pozostałości po nieudanej próbie (bez G/D) - inaczej stare próbki
        # z sliced_audios/ zmieszałyby się z nowym datasetem.
        shutil.rmtree(model_dir)

    reuse_data = resume and _has_extracted_dataset(model_dir)
    if has_upload and not reuse_data and model_dir.exists():
        shutil.rmtree(model_dir)
    if resume and not has_upload and existing_g is None:
        raise ApplioTrainingError(
            f"Brak checkpointu do wznowienia modelu „{name}” w tej sesji. "
            "W nowej sesji Colaba wgraj te same próbki oraz pliki G i D "
            "zapisane wcześniej na Dysku Google."
        )
    if not reuse_data and not sample_paths:
        raise ApplioTrainingError(
            "Wgraj próbki treningowe. Przy wznowieniu w nowej sesji - te same, "
            "co w pierwszym treningu."
        )
    if reuse_data:
        dir_sr = _model_dir_sample_rate(model_dir)
        if dir_sr and dir_sr != sample_rate:
            raise ApplioTrainingError(
                f"Model „{name}” był trenowany z {dir_sr} Hz - ustaw tę samą "
                f"częstotliwość próbkowania, teraz jest {sample_rate} Hz."
            )

    resume_epoch: int | None = None
    if resume:
        ckpt_to_check = Path(resume_g_path) if has_upload else existing_g
        resume_epoch = read_checkpoint_epoch(config, ckpt_to_check)
        if has_upload and resume_epoch is None:
            raise ApplioTrainingError(
                f"Plik {Path(resume_g_path).name} nie wygląda na checkpoint G_ "
                "z Applio - nie da się odczytać epoki."
            )
        if resume_epoch is not None and epochs <= resume_epoch:
            raise ApplioTrainingError(
                f"Checkpoint jest z epoki {resume_epoch} - ustaw łączną liczbę "
                f"epok większą niż {resume_epoch}, np. {resume_epoch + 100}, "
                "w trybie Zaawansowany → Epoki."
            )

    monitor = monitor or TrainingMonitor(name)
    backup_note = ""
    backup_dir = None
    if auto_backup:
        backup_dir, error = drive_dest_dir(f"{name}/checkpoints")
        backup_note = (
            f"Auto-kopia G/D na Dysk Google: {backup_dir}\n"
            if backup_dir
            else f"Auto-kopia G/D wyłączona: {error}\n"
        )
        if backup_dir is None:
            save_log.record(f"Trening „{name}”: auto-kopia G/D wyłączona. {error}")
    monitor.configure(model_dir, epochs, save_every, backup_dir, resume_epoch)

    device_info = probe_torch_device(config.python_bin)
    gpu_arg = "0" if "GPU" in device_info else "-"

    yield (
        "start",
        f"{device_info}\nNazwa modelu: {name}\n{backup_note}"
        + (
            ""
            if gpu_arg == "0"
            else "\nUwaga: brak GPU - trening na CPU jest bardzo wolny/niepraktyczny "
            "poza krótkim testem; docelowo używaj GPU w Colabie.\n"
        ),
        None,
    )

    work_dir = Path(tempfile.mkdtemp(prefix="rvc_train_"))
    timeout_s = int(os.environ.get("APPLIO_TRAIN_TIMEOUT_S", "43200"))

    try:
        if reuse_data:
            yield (
                "preprocess",
                "Krok 1-2/3: pominięte - dane z poprzedniego treningu tego modelu "
                "są już przetworzone w tej sesji.\n",
                None,
            )
        else:
            dataset_dir = _prepare_dataset_dir(sample_paths, work_dir)

            monitor.phase = "Krok 1/3: wstępne przetwarzanie próbek…"
            yield "preprocess", f"{monitor.phase}\n", None
            preprocess_cmd = [
                config.python_bin,
                "core.py",
                "preprocess",
                "--model-name", name,
                "--dataset-path", str(dataset_dir),
                "--sample-rate", str(sample_rate),
                "--cut-preprocess", "Automatic",
            ]
            for partial in _stream_step(
                preprocess_cmd, cwd=config.applio_dir, timeout_s=timeout_s,
                step_name="preprocess", monitor=monitor,
            ):
                yield "preprocess", f"Krok 1/3: wstępne przetwarzanie próbek…\n\n{partial}", None

            monitor.phase = "Krok 2/3: ekstrakcja cech głosu…"
            yield "extract", f"{monitor.phase}\n", None
            extract_cmd = [
                config.python_bin,
                "core.py",
                "extract",
                "--model-name", name,
                "--f0-method", method,
                "--sample-rate", str(sample_rate),
                "--embedder-model", embedder_model,
                "--gpu", gpu_arg,
            ]
            for partial in _stream_step(
                extract_cmd, cwd=config.applio_dir, timeout_s=timeout_s,
                step_name="extract", monitor=monitor,
            ):
                yield "extract", f"Krok 2/3: ekstrakcja cech…\n\n{partial}", None

        if has_upload:
            _install_uploaded_checkpoints(model_dir, resume_g_path, resume_d_path)

        if resume:
            start_info = (
                f"wznowienie od epoki {resume_epoch + 1}"
                if resume_epoch is not None
                else "wznowienie od ostatniego checkpointu"
            )
        else:
            start_info = "nowy trening"
        # Applio zapisuje G/D tylko w epokach podzielnych przez save_every -
        # końcowa epoka spoza tej siatki nie trafi do checkpointu.
        grid_note = (
            ""
            if epochs % save_every == 0
            else (
                f"\nUwaga: {epochs} nie dzieli się przez {save_every} - ostatni "
                f"checkpoint G/D będzie z epoki {epochs - epochs % save_every}, "
                "a nie z końcowej - od niej zacznie się dalsze wznowienie."
            )
        )
        train_header = (
            f"Krok 3/3: trening - {start_info}, cel: {epochs} epok, paczka "
            f"{batch_size}, checkpoint co {save_every} epok.{grid_note}"
        )
        monitor.phase = "Krok 3/3: uruchamianie treningu, ładowanie modelu i danych…"
        monitor.training_started = True
        yield "train", f"{train_header}\nTo może zająć długo, cierpliwości…\n", None
        train_cmd = [
            config.python_bin,
            "core.py",
            "train",
            "--model-name", name,
            "--save-every-epoch", str(save_every),
            "--save-only-latest",  # jedna para G_/D_ nadpisywana przy każdym zapisie
            # Mid-run extract wyłączony patchiem Colab (_patch_applio_train_for_colab);
            # mały model powstaje na końcu (Final model + nasz fallback).
            "--total-epoch", str(epochs),
            "--sample-rate", str(sample_rate),
            "--batch-size", str(int(batch_size)),
            "--gpu", gpu_arg,
        ]
        for partial in _stream_step(
            train_cmd, cwd=config.applio_dir, timeout_s=timeout_s,
            step_name="train", monitor=monitor,
        ):
            yield "train", f"{train_header}\n\n{partial}", None
    except TrainingStopped:
        yield from _package_after_stop(config, monitor, model_dir, name, work_dir, timeout_s)
        return

    monitor.phase = "Pakowanie modelu…"
    yield "done", "Trening zakończony - pakowanie modelu…", None
    pth, index = _pick_final_model(config, model_dir, name)
    zip_path = _zip_model(work_dir, name, pth, index)
    yield from _wait_backups_live(monitor, "Trening zakończony.", timeout_s=600)
    yield "done", "Trening zakończony - zapisuję model na Dysku Google…", None
    drive_note = _auto_save_model(zip_path, name)

    summary = (
        f"Gotowe! Model: {pth.name}"
        + (f" + {index.name}" if index else ". Brak pliku .index - sprawdź log ekstrakcji.")
        + f"\n{drive_note.split('  ', 1)[-1]}"
        + "\nPobieranie i ponowny zapis na Dysk: krok 5 poniżej."
    )
    yield "done", summary, str(zip_path)


def _zip_model(work_dir: Path, name: str, pth: Path, index: Path | None) -> Path:
    zip_path = work_dir / f"{pth.stem if pth.stem.startswith(name) else name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pth, pth.name)
        if index is not None:
            zf.write(index, index.name)
    return zip_path


def _wait_backups_live(
    monitor: TrainingMonitor, header: str, timeout_s: float
) -> Iterator[tuple[str, str, str | None]]:
    """Czeka na auto-kopię G/D na Drive, odświeżając UI (nie blokuje go minutami)."""
    deadline = time.monotonic() + timeout_s
    while monitor.backups_pending() and time.monotonic() < deadline:
        yield "backup", f"{header}\n{monitor.backup_status or 'Kończę kopię checkpointu na Dysk Google…'}", None
        time.sleep(1)


def _package_after_stop(
    config: ApplioConfig,
    monitor: TrainingMonitor,
    model_dir: Path,
    name: str,
    work_dir: Path,
    timeout_s: int,
) -> Iterator[tuple[str, str, str | None]]:
    """Po „Zatrzymaj trening”: ZIP z modelu z ostatniego checkpointu (+ świeży .index)."""
    yield from _wait_backups_live(monitor, "Trening zatrzymany.", timeout_s=300)
    last = monitor.last_ckpt_epoch
    resume_hint = (
        f"Ostatni checkpoint G/D: epoka {last} - zaznacz „Wznów trening”, "
        "żeby kontynuować."
        if last
        else "Brak checkpointu G/D - zatrzymano przed pierwszym zapisem."
    )
    has_model = _list_infer_pths(model_dir) or _list_generator_ckpts(model_dir)
    if not monitor.training_started or not has_model:
        yield "stopped", f"Trening zatrzymany.\n{resume_hint}", None
        return

    monitor.phase = "Zatrzymano - buduję .index dla modelu z ostatniego checkpointu…"
    yield "stopped", f"Trening zatrzymany.\n{resume_hint}\n\n{monitor.phase}", None
    index_note = ""
    try:
        for partial in _stream_step(
            [config.python_bin, "core.py", "index", "--model-name", name],
            cwd=config.applio_dir, timeout_s=timeout_s, step_name="index",
        ):
            yield "stopped", f"Trening zatrzymany.\n{resume_hint}\n\n{partial}", None
    except ApplioTrainingError as exc:
        index_note = f"\nPlik .index nie powstał: {str(exc).splitlines()[0]}"

    pth, index = _pick_final_model(config, model_dir, name)
    zip_path = _zip_model(work_dir, name, pth, index)
    yield "stopped", "Trening zatrzymany - zapisuję model na Dysku Google…", None
    drive_note = _auto_save_model(zip_path, name)
    summary = (
        f"Trening zatrzymany. Model z ostatniego zapisu: {pth.name}"
        + (f" + {index.name}" if index else "")
        + index_note
        + f"\n{drive_note.split('  ', 1)[-1]}\n{resume_hint}"
        + "\nPobieranie i ponowny zapis na Dysk: krok 5 poniżej."
    )
    yield "stopped", summary, str(zip_path)


def save_model_to_drive(zip_path: str | None, model_name: str) -> str:
    """Kopiuje gotowy ZIP modelu (.pth + .index) do ``MyDrive/RVC_Colab_Models/``.

    Każda próba (automatyczna po treningu i ręczne ponowienie) trafia do
    historii zapisów (:mod:`rvc_pipeline.save_log`).
    """
    if not zip_path or not Path(zip_path).is_file():
        return save_log.record(
            "Dysk Google: brak gotowego modelu ZIP - najpierw dokończ trening."
        )
    dest_dir, error = drive_dest_dir()
    if dest_dir is None:
        return save_log.record(f"Model nie został zapisany na Dysku. {error}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        dest_path = dest_dir / f"{sanitize_model_name(model_name)}_{timestamp}.zip"
        copy_atomic(Path(zip_path), dest_path)
    except (OSError, ApplioTrainingError) as exc:
        return save_log.record(f"Dysk Google: nie udało się zapisać modelu: {exc}")

    return save_log.record(f"Dysk Google: zapisano model w {dest_path}")


def retry_model_save(zip_path: str | None, model_name: str) -> str:
    """„Zapisz model na Dysku ponownie”: ZIP ze strony albo odbudowany z ``logs/<nazwa>``.

    Żeby podłączyć Dysk, trzeba zrestartować komórkę UI - strona traci wtedy
    ścieżkę ZIP-a, ale gotowy model w ``/content`` zostaje.
    """
    if zip_path and Path(zip_path).is_file():
        return save_model_to_drive(zip_path, model_name)
    try:
        config = ApplioConfig.from_env()
        name = sanitize_model_name(model_name)
    except (ApplioConfigError, ApplioTrainingError) as exc:
        return save_log.record(f"Model nie został zapisany na Dysku. {exc}")
    model_dir = config.applio_dir / "logs" / name
    infer_pths = _list_infer_pths(model_dir) if model_dir.is_dir() else []
    if not infer_pths:
        return save_log.record(
            f"Dysk Google: brak gotowego modelu „{name}” w tej sesji - "
            "najpierw dokończ trening."
        )
    pth = max(infer_pths, key=lambda p: p.stat().st_mtime)
    indexes = sorted(model_dir.rglob("*.index"), key=lambda p: p.stat().st_mtime)
    rebuilt = _zip_model(
        Path(tempfile.mkdtemp(prefix="rvc_retry_")), name, pth, indexes[-1] if indexes else None
    )
    return save_model_to_drive(str(rebuilt), name)


def _auto_save_model(zip_path: Path, name: str) -> str:
    if not drive_mounted():
        return save_log.record(
            f"Model „{name}” zapisany tylko lokalnie - Dysk Google nie jest "
            "podłączony. Pobierz ZIP albo podłącz Dysk i kliknij „Zapisz model "
            "na Dysku ponownie”."
        )
    return save_model_to_drive(str(zip_path), name)


def prepare_checkpoint_downloads(model_name: str, epoch: int | None) -> tuple[str | None, str | None]:
    """Najnowsze G i D pod czytelną nazwą (``G_<model>_e<epoka>.pth``) do pobrania.

    Twardy link zamiast kopii (setki MB) - Gradio i tak kopiuje plik do swojej
    pamięci podręcznej przy wysyłce do przeglądarki.
    """
    try:
        config = ApplioConfig.from_env()
        name = sanitize_model_name(model_name)
    except (ApplioConfigError, ApplioTrainingError):
        return None, None
    model_dir = config.applio_dir / "logs" / name
    out_dir = config.applio_dir / "rvc_downloads" / name
    result: list[str | None] = []
    for kind in ("G", "D"):
        src = latest_training_checkpoint(model_dir, kind)
        if src is None:
            result.append(None)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{kind}_*.pth"):
            old.unlink(missing_ok=True)
        dest = out_dir / checkpoint_drive_name(kind, name, epoch)
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
        result.append(str(dest))
    return result[0], result[1]


def save_checkpoint_to_drive(model_name: str, kind: str) -> str:
    """Kopiuje najnowszy checkpoint treningowy ``G`` albo ``D`` na Drive.

    Para G + D z tej samej epoki pozwala wznowić trening w nowej sesji
    Colaba. Działa też w trakcie treningu (kopiuje ostatni zapisany
    checkpoint; jeśli Applio akurat go zapisuje - najpierw czeka).
    Nazwa ``G_<model>_e<epoka>.pth`` pozwala dobrać parę z D.
    """
    kind = kind.upper()
    if kind not in ("G", "D"):
        raise ValueError(f"kind musi być 'G' albo 'D', jest: {kind!r}")
    label = "generatora (G)" if kind == "G" else "dyskryminatora (D)"
    try:
        name = sanitize_model_name(model_name)
    except ApplioTrainingError as exc:
        return f"Google Drive: {exc}"
    try:
        config = ApplioConfig.from_env()
    except ApplioConfigError as exc:
        return f"Błąd konfiguracji środowiska Applio:\n{exc}"

    model_dir = config.applio_dir / "logs" / name
    monitor = get_active(name)
    if monitor is not None and monitor._save_ts:
        wait_for_checkpoint_written(model_dir, monitor._save_ts)

    ckpt = latest_training_checkpoint(model_dir, kind)
    if ckpt is None:
        return (
            f"Brak checkpointu {label} dla modelu „{name}” w tej sesji - "
            "pojawi się po pierwszym zapisie (co N epok z suwaka)."
        )

    dest_dir, error = drive_dest_dir(f"{name}/checkpoints")
    if dest_dir is None:
        return error

    epoch = read_checkpoint_epoch(config, ckpt)
    dest_path = dest_dir / checkpoint_drive_name(kind, name, epoch)
    try:
        copy_atomic(ckpt, dest_path)
    except OSError as exc:
        return f"Google Drive: nie udało się zapisać checkpointu {label} ({exc})."

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    epoch_info = f" (epoka {epoch})" if epoch is not None else ""
    return (
        f"Google Drive: zapisano checkpoint {label}{epoch_info}, "
        f"{size_mb:.0f} MB → {dest_path}"
    )


def list_drive_checkpoints(model_name: str) -> tuple[list[CheckpointPair], str]:
    """Pary G/D na Drive dla modelu (od najnowszej epoki) + komunikat dla UI."""
    try:
        name = sanitize_model_name(model_name)
    except ApplioTrainingError:
        return [], "Podaj nazwę modelu, żeby wyszukać jego checkpointy na Drive."
    root, error = drive_dest_dir()
    if root is None:
        return [], error
    ckpt_dir = drive_checkpoint_dir(name)
    pairs = list_checkpoint_pairs(ckpt_dir)
    if not pairs:
        return [], (
            f"Brak par G/D w {ckpt_dir} - zapisz je przyciskami „Zapisz "
            "checkpoint G/D” albo włącz auto-kopię w poprzedniej sesji."
        )
    return pairs, f"Znaleziono {len(pairs)} par(y) G/D w {ckpt_dir}."
