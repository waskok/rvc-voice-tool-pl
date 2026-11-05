"""Cienki wrapper na silnik Applio (RVC) - Zakładka 3 (inferencja).

Applio ma własne zależności (m.in. `gradio==6.20.0`, `torch==2.11.0`),
niekompatybilne z naszym UI (`gradio>=5.7,<6`). Dlatego Applio **nie jest
importowane** do tego procesu - jest wołane jako osobny proces (subprocess)
z osobnym interpreterem Pythona / venv, w którym zainstalowano zależności
Applio. Dzięki temu wersje Gradio się nie gryzą (patrz `documents/POROWNANIA_I_ANALIZY.md`).

Konfiguracja przez zmienne środowiskowe (patrz `SETUP_APPLIO.md`):
    APPLIO_DIR     - ścieżka do klonu repo IAHispano/Applio (tag zalecany: 3.6.5)
    APPLIO_PYTHON  - ścieżka do interpretera Pythona z zależnościami Applio
                     (osobny venv niż ten, w którym działa nasze UI)
    APPLIO_TIMEOUT_S - opcjonalny limit czasu procesu inferencji (domyślnie 1800 s)

Źródła:
  - CLI Applio (`core.py infer`, argumenty): https://docs.applio.org/reference/cli/
  - Repo / tag: https://github.com/IAHispano/Applio (LICENSE: MIT)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


class ApplioConfigError(RuntimeError):
    """Brak lub błędna konfiguracja środowiska Applio (APPLIO_DIR / APPLIO_PYTHON)."""


class ApplioInferenceError(RuntimeError):
    """Applio `core.py infer` zwrócił błąd albo nie wyprodukował pliku wynikowego."""


# Metody F0 realnie działające w Applio 3.6.5 (`get_f0` w rvc/infer/pipeline.py).
# CLI Click *wymienia* też hybrid[…], ale w tagu 3.6.5 hybrid NIE jest
# zaimplementowany → UnboundLocalError: f0. Dlatego hybridów tu nie oferujemy.
F0_METHODS: tuple[str, ...] = (
    "rmvpe",
    "fcpe",
    "crepe-tiny",
    "crepe",
)
DEFAULT_F0_METHOD = "rmvpe"

# Te same nazwy co `--embedder-model` w Applio 3.6.5 (core.py infer / extract).
# Bez "custom" - wymagałoby ścieżki do własnego modelu, poza zakresem narzędzia.
EMBEDDER_MODELS: tuple[str, ...] = (
    "contentvec",
    "spin",
    "spin-v2",
    "chinese-hubert-base",
    "japanese-hubert-base",
    "korean-hubert-base",
)
DEFAULT_EMBEDDER_MODEL = "contentvec"
# Wartość listy w zakładce 3: nie wysyłamy jej do Applio, tylko czytamy
# pole `embedder_model` zapisane w pliku .pth przy eksporcie modelu.
EMBEDDER_FROM_MODEL = "z modelu"

# `--index-rate` w Applio to FloatRange(0, 1). Domyślne 0,75 (nie 0,3 z CLI),
# bo przy indeksie barwa jest wtedy bliższa próbkom treningowym.
DEFAULT_INDEX_RATE = 0.75


def probe_torch_device(python_bin: str | None = None) -> str:
    """Sprawdza, czy venv Applio widzi CUDA (GPU) czy tylko CPU.

    Woła ``APPLIO_PYTHON`` (osobny interpreter) - w naszym UI Gradio nie ma
    torcha, więc import lokalny nic by nie dał.
    """
    bin_path = (python_bin or os.environ.get("APPLIO_PYTHON", "")).strip()
    if not bin_path:
        return "urządzenie: nieznane, brak APPLIO_PYTHON"
    try:
        proc = subprocess.run(
            [
                bin_path,
                "-c",
                "import torch; "
                "print('cuda' if torch.cuda.is_available() else 'cpu'); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "urządzenie: nie udało się sprawdzić"

    if proc.returncode != 0:
        return "urządzenie: nie udało się sprawdzić"

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    kind = lines[0] if lines else "unknown"
    name = lines[1] if len(lines) > 1 else ""
    if kind == "cuda":
        return f"urządzenie: GPU{f' - {name}' if name else ''}"
    if kind == "cpu":
        return "urządzenie: CPU - brak GPU, konwersja będzie wolniejsza"
    return f"urządzenie: {kind}"


@dataclass(frozen=True)
class ApplioConfig:
    applio_dir: Path
    python_bin: str

    @classmethod
    def from_env(cls) -> "ApplioConfig":
        applio_dir = os.environ.get("APPLIO_DIR", "").strip()
        python_bin = os.environ.get("APPLIO_PYTHON", "").strip()

        if not applio_dir:
            raise ApplioConfigError(
                "Brak zmiennej środowiskowej APPLIO_DIR - ustaw ją na ścieżkę "
                "do klonu repo IAHispano/Applio (np. export APPLIO_DIR=/opt/Applio). "
                "Instrukcja: SETUP_APPLIO.md."
            )
        if not python_bin:
            raise ApplioConfigError(
                "Brak zmiennej środowiskowej APPLIO_PYTHON - ustaw ją na ścieżkę "
                "do interpretera Pythona z zainstalowanymi zależnościami Applio "
                "(osobny venv niż ten dla naszego UI, bo Applio wymaga Gradio 6). "
                "Instrukcja: SETUP_APPLIO.md."
            )

        applio_path = Path(applio_dir).expanduser()
        core_py = applio_path / "core.py"
        if not core_py.is_file():
            raise ApplioConfigError(
                f"Nie znaleziono {core_py} - sprawdź, czy APPLIO_DIR wskazuje "
                "na katalog z klonem repo Applio (musi zawierać core.py)."
            )

        python_path = Path(python_bin).expanduser()
        if not python_path.is_file() and shutil.which(python_bin) is None:
            raise ApplioConfigError(
                f"Nie znaleziono interpretera Pythona: {python_bin} - sprawdź "
                "APPLIO_PYTHON (ścieżka do python w venv Applio, np. "
                ".../applio-venv/bin/python)."
            )

        return cls(applio_dir=applio_path, python_bin=str(python_bin))


def _pick_inference_pth(candidates: list[Path]) -> tuple[Path, list[str]]:
    """Wybiera plik `.pth` nadający się do inferencji.

    Do ``core.py infer`` nadaje się tylko **eksportowany** model (ma klucz
    ``config`` w checkpointcie) - zwykle ``{nazwa}_*e_*s.pth`` albo plik bez
    prefiksu ``G_``/``D_``.

    ``G_*.pth`` / ``D_*.pth`` to checkpointy treningowe - inferencja pada na
    ``KeyError: 'config'``. Nie używamy ich jako fallbacku.
    """
    warnings: list[str] = []
    if not candidates:
        raise ApplioInferenceError(
            "Nie znaleziono pliku .pth - wgraj model jako .zip lub .pth."
        )

    names_lower = [(p, p.name.lower()) for p in candidates]
    infer = [p for p, n in names_lower if not (n.startswith("d_") or n.startswith("g_"))]
    generators = [p for p, n in names_lower if n.startswith("g_")]
    discriminators = [p for p, n in names_lower if n.startswith("d_")]

    if infer:
        chosen = max(infer, key=lambda p: p.stat().st_mtime)
        if len(candidates) > 1:
            warnings.append(
                f"W ZIP znaleziono {len(candidates)} plików .pth - użyto "
                f"modelu inferencyjnego: {chosen.name}."
            )
        return chosen, warnings

    if generators:
        g_names = ", ".join(p.name for p in generators)
        raise ApplioInferenceError(
            "W ZIP-ie / wśród wgranych plików jest tylko checkpoint treningowy "
            f"(G_*.pth): {g_names}.\n"
            "To nie jest model do konwersji - Applio wymaga małego pliku .pth "
            "z kluczem „config” (zwykle `{nazwa}_Xe_Ys.pth` z katalogu "
            "`logs/<model>/` po treningu).\n"
            "W Colabie: zbuduj go z G_ (komórka recovery) albo uruchom ponownie "
            "trening na poprawionej Zakładce 2, która pakuje właściwy plik."
        )

    names = ", ".join(p.name for p in discriminators)
    raise ApplioInferenceError(
        "W ZIP-ie / wśród wgranych plików są tylko dyskryminatory "
        f"(D_*.pth): {names}. Do konwersji potrzebny jest eksportowany model "
        ".pth (bez prefiksu D_/G_), nie checkpoint treningowy."
    )


def _extract_model_files(
    model_path: str, index_path: str | None, work_dir: Path
) -> tuple[Path, Path | None, list[str]]:
    """Ustala pliki `.pth` (wymagany) i `.index` (opcjonalny) z uploadu.

    Obsługuje dwa warianty wejścia:
      - ZIP z modelem (szuka `*.pth` / `*.index` w archiwum),
      - pojedynczy plik `.pth` (+ opcjonalnie osobny `.index`).
    """
    warnings: list[str] = []
    model_p = Path(model_path)
    suffix = model_p.suffix.lower()

    if suffix == ".zip":
        extract_dir = work_dir / "model"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(model_p) as zf:
            zf.extractall(extract_dir)

        pth_candidates = sorted(extract_dir.rglob("*.pth"))
        index_candidates = sorted(extract_dir.rglob("*.index"))

        if not pth_candidates:
            raise ApplioInferenceError(
                "W ZIP-ie nie znaleziono pliku .pth - sprawdź zawartość archiwum "
                "(powinien być eksportowany model .pth bez prefiksu G_/D_, "
                "opcjonalnie .index)."
            )
        pth, pick_warnings = _pick_inference_pth(pth_candidates)
        warnings.extend(pick_warnings)

        index: Path | None = (
            max(index_candidates, key=lambda p: p.stat().st_mtime)
            if index_candidates
            else None
        )
        if len(index_candidates) > 1:
            warnings.append(
                f"Uwaga: w ZIP znaleziono {len(index_candidates)} plików .index - "
                f"użyto najnowszego: {index.name}."
            )
    elif suffix == ".pth":
        low = model_p.name.lower()
        if low.startswith("d_"):
            raise ApplioInferenceError(
                f"Wgrany plik {model_p.name} to dyskryminator (D_*.pth) - "
                "nie nadaje się do konwersji. Wgraj eksportowany model .pth "
                "(bez prefiksu D_/G_)."
            )
        if low.startswith("g_"):
            raise ApplioInferenceError(
                f"Wgrany plik {model_p.name} to checkpoint treningowy (G_*.pth) - "
                "nie nadaje się do konwersji (Applio: KeyError „config”).\n"
                "Potrzebujesz małego modelu `{nazwa}_Xe_Ys.pth` z "
                "`/content/Applio/logs/<model>/` albo zbuduj go z G_ komórką recovery."
            )
        pth = model_p
        index = Path(index_path).expanduser() if index_path else None
    else:
        raise ApplioInferenceError(
            f"Nieobsługiwany format pliku modelu: '{model_p.suffix}'. "
            "Wgraj model jako .zip (pth + index) albo .pth."
        )

    if index is None:
        warnings.append(
            "Brak pliku .index - konwersja zostanie wykonana bez wyszukiwania "
            "wektorowego (bez --index-path). Jakość / barwa mogą być gorsze "
            "niż z indeksem."
        )
    elif not index.is_file():
        warnings.append(
            f"Podany plik .index nie istnieje: {index} - kontynuuję bez indeksu."
        )
        index = None

    return pth, index, warnings


def normalize_index_rate(value: float | int | str | None) -> float:
    """Udział indeksu z suwaka do ``--index-rate`` (0-1)."""
    if value is None or value == "":
        return DEFAULT_INDEX_RATE
    try:
        rate = round(float(value), 2)
    except (TypeError, ValueError) as exc:
        raise ApplioInferenceError(
            "Udział indeksu musi być liczbą od 0 do 1."
        ) from exc
    if rate < 0 or rate > 1:
        raise ApplioInferenceError("Udział indeksu musi być liczbą od 0 do 1.")
    return rate


def format_index_rate(rate: float) -> str:
    """``0.75`` → ``0.75``, ``1.0`` → ``1`` - Click przyjmuje obie postaci."""
    text = f"{rate:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def resolve_embedder(selected: str | None, stored: str | None) -> tuple[str, str]:
    """Ustala ``--embedder-model`` i zdanie do dziennika konwersji.

    ``selected`` to wartość z zakładki 3. ``EMBEDDER_FROM_MODEL`` (albo puste)
    bierze nazwę z pliku .pth, czyli wybór z treningu. Jawny wybór z listy
    ma pierwszeństwo.
    """
    choice = (selected or EMBEDDER_FROM_MODEL).strip().lower()
    if choice in ("", EMBEDDER_FROM_MODEL):
        stored_name = (stored or "").strip().lower()
        if stored_name in EMBEDDER_MODELS:
            return stored_name, (
                f"model cech: {stored_name} "
                "(zapisany w pliku modelu przy treningu)"
            )
        if stored_name:
            return DEFAULT_EMBEDDER_MODEL, (
                f"W pliku modelu jest nieznany model cech „{stored}”. "
                f"Używam {DEFAULT_EMBEDDER_MODEL}."
            )
        return DEFAULT_EMBEDDER_MODEL, (
            "W pliku modelu nie ma zapisanego modelu cech. "
            f"Używam {DEFAULT_EMBEDDER_MODEL}."
        )
    if choice in EMBEDDER_MODELS:
        return choice, f"model cech: {choice} (wybrany w zakładce 3)"
    allowed = ", ".join((EMBEDDER_FROM_MODEL, *EMBEDDER_MODELS))
    raise ApplioInferenceError(
        f"Nieznany model cech: '{selected}'. Dozwolone: {allowed}."
    )


def read_embedder_from_pth(python_bin: str, pth: Path) -> str | None:
    """Czyta pole ``embedder_model`` z .pth interpreterem Applio (tam jest torch).

    Proces interfejsu nie ma biblioteki torch. Przy błędzie odczytu zwraca
    ``None`` - konwersja i tak ruszy, z domyślnym modelem cech i wpisem w dzienniku.
    """
    script = (
        "import sys\n"
        "import torch\n"
        "path = sys.argv[1]\n"
        "try:\n"
        "    ckpt = torch.load(path, map_location='cpu', weights_only=False)\n"
        "except TypeError:\n"
        "    ckpt = torch.load(path, map_location='cpu')\n"
        "value = ''\n"
        "if isinstance(ckpt, dict):\n"
        "    raw = ckpt.get('embedder_model')\n"
        "    if raw:\n"
        "        value = str(raw)\n"
        "print(value)\n"
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", script, str(pth)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines[-1] if lines else None


def run_inference(
    model_path: str,
    source_audio: str,
    pitch: int = 0,
    index_path: str | None = None,
    f0_method: str = "rmvpe",
    index_rate: float = DEFAULT_INDEX_RATE,
    embedder_model: str = EMBEDDER_FROM_MODEL,
) -> tuple[str, str]:
    """Uruchamia konwersję głosu przez `python core.py infer` (Applio).

    Zwraca ``(ścieżka_do_wav_wyniku, log_tekstowy)``.

    Rzuca:
      - ``ApplioConfigError`` - gdy brak / błędna konfiguracja środowiska,
      - ``ApplioInferenceError`` - gdy proces Applio zwrócił błąd albo nie
        wyprodukował pliku wynikowego.

    Obie klasy wyjątków niosą komunikat po polsku gotowy do pokazania w UI.
    """
    config = ApplioConfig.from_env()

    if not source_audio:
        raise ApplioInferenceError("Brak nagrania źródłowego do konwersji.")

    method = (f0_method or DEFAULT_F0_METHOD).strip().lower()
    if method.startswith("hybrid"):
        raise ApplioInferenceError(
            f"Metoda F0 '{f0_method}' nie działa w Applio 3.6.5 (błąd silnika: "
            "hybrid jest w CLI, ale nie w get_f0). Wybierz: "
            f"{', '.join(F0_METHODS)}."
        )
    if method not in F0_METHODS:
        raise ApplioInferenceError(
            f"Nieznana metoda F0: '{f0_method}'. Dozwolone: {', '.join(F0_METHODS)}."
        )

    rate = normalize_index_rate(index_rate)
    wants_stored_embedder = (embedder_model or EMBEDDER_FROM_MODEL).strip().lower() in (
        "",
        EMBEDDER_FROM_MODEL,
    )

    device_info = probe_torch_device(config.python_bin)

    work_dir = Path(tempfile.mkdtemp(prefix="rvc_infer_"))
    pth, index, warnings = _extract_model_files(model_path, index_path, work_dir)

    output_path = work_dir / "output.wav"

    stored_embedder = (
        read_embedder_from_pth(config.python_bin, pth) if wants_stored_embedder else None
    )
    embedder, embedder_note = resolve_embedder(embedder_model, stored_embedder)

    # Applio 3.6.5 (Click) używa kebab-case: --input-path, nie --input_path.
    # Potwierdzone: `core.py infer --help` / błąd "No such option: --input_path".
    # Bez --embedder-model CLI Applio wstawia contentvec, także gdy trening
    # szedł innym modelem cech zapisanym w pliku .pth.
    cmd = [
        config.python_bin,
        "core.py",
        "infer",
        "--input-path", str(source_audio),
        "--output-path", str(output_path),
        "--pth-path", str(pth),
        "--pitch", str(int(pitch)),
        "--f0-method", method,
        "--embedder-model", embedder,
        "--export-format", "WAV",
    ]
    if index is not None:
        cmd += ["--index-path", str(index), "--index-rate", format_index_rate(rate)]

    timeout_s = int(os.environ.get("APPLIO_TIMEOUT_S", "1800"))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(config.applio_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ApplioInferenceError(
            f"Przekroczono limit czasu inferencji ({timeout_s} s). "
            "Spróbuj krótszego nagrania albo zwiększ APPLIO_TIMEOUT_S."
        ) from exc
    except OSError as exc:
        raise ApplioConfigError(
            f"Nie udało się uruchomić {config.python_bin} - sprawdź APPLIO_PYTHON. "
            f"Błąd systemowy: {exc}"
        ) from exc

    log_lines = [device_info, f"metoda F0: {method}", embedder_note]
    if index is not None:
        log_lines.append(f"udział indeksu: {format_index_rate(rate)}")
    else:
        log_lines.append("udział indeksu: pominięty, brak pliku .index")
    log_lines.extend(warnings)
    log_lines.append(f"Polecenie: {' '.join(cmd)}")
    if proc.stdout.strip():
        log_lines.append("--- stdout (Applio) ---")
        log_lines.append(proc.stdout.strip())
    if proc.stderr.strip():
        log_lines.append("--- stderr (Applio) ---")
        log_lines.append(proc.stderr.strip())

    if proc.returncode != 0:
        hint = ""
        if proc.returncode < 0:
            hint = (
                f"\n\nProces Applio został zabity przez system "
                f"(sygnał {-proc.returncode}). Na darmowym Colabie przy limicie "
                "zasobów / runtime CPU to częste - spróbuj GPU albo krótszego nagrania."
            )
        raise ApplioInferenceError(
            f"Applio core.py zakończył się błędem (kod {proc.returncode}). "
            "Szczegóły w logu poniżej."
            f"{hint}\n\n" + "\n".join(log_lines)
        )

    if not output_path.is_file():
        raise ApplioInferenceError(
            "core.py zakończył się bez błędu, ale nie znaleziono pliku "
            "wynikowego. Szczegóły w logu poniżej.\n\n" + "\n".join(log_lines)
        )

    return str(output_path), "\n".join(log_lines)
