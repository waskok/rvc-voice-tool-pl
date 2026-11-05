"""Narzędzia audio - Zakładka 1 (Faza 2): realna obróbka pod dataset RVC.

Kroki wykonywane w `prepare_audio`:
  1. Wczytanie + zamiana na mono + resampling do docelowej częstotliwości
     (RVC wymaga mono; różne wersje modeli oczekują stałego SR, np. 40k/48k).
  2. Cięcie na fragmenty rozdzielone ciszą (auto-segmentacja; opcjonalne).
  3. Filtr długości - za krótkie odrzucane (oddechy, kliknięcia); za długie
     dzielone na stałe okna o długości max (zamiast odrzucania całego pliku).
  4. Normalizacja głośności do docelowego poziomu LUFS (EBU R128) - ujednolica
     głośność próbek niezależnie od tego, jak nagrywał je użytkownik.
  5. Odszumianie (spectral gating) - ostrożne ustawienia domyślne, żeby nie
     wprowadzać sztucznych artefaktów w miejsce szumu.
  6. Podsumowanie jakości datasetu (łączny czas, liczba próbek, ostrzeżenie).

Biblioteki (wszystkie CPU, permisywne licencje):
  - librosa (ISC)      - wczytanie/mono/resampling/cięcie po ciszy
  - pyloudnorm (MIT)   - normalizacja głośności (ITU-R BS.1770 / EBU R128)
  - noisereduce (MIT)  - odszumianie
"""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# Częstotliwości próbkowania używane przez popularne wersje modeli RVC.
TARGET_SAMPLE_RATES: list[int] = [32000, 40000, 48000]
DEFAULT_SAMPLE_RATE = 40000

# Docelowy poziom głośności (LUFS) - -23 LUFS to standard EBU R128
# (nadawczy punkt odniesienia); dla mowy/śpiewu solo to bezpieczna wartość
# bez przycinania (clippingu) i bez zbyt cichego sygnału.
TARGET_LOUDNESS_LUFS = -23.0


@dataclass
class DatasetSummary:
    """Podsumowanie jednego przebiegu przygotowania datasetu."""

    accepted: int = 0
    rejected_short: int = 0
    # Ile próbek powstało z podziału fragmentów dłuższych niż max (nie są odrzucane).
    resplit_long: int = 0
    total_duration_s: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    warnings: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        minutes = self.total_duration_s / 60.0
        quality = (
            "Dobrze - powyżej 10 minut czystego materiału."
            if minutes >= 10
            else "Mało - poniżej 5 minut, model może wyjść niestabilny."
            if minutes < 5
            else "OK, ale więcej danych zwykle poprawia jakość modelu."
        )
        lines = [
            f"Zaakceptowane próbki: {self.accepted}",
            f"Łączny czas czystego audio: {minutes:.1f} min",
            f"Ocena ilości danych: {quality}",
            f"Odrzucone jako za krótkie: {self.rejected_short}",
            f"Podzielone z za długich: {self.resplit_long}",
            f"Częstotliwość próbkowania: {self.sample_rate} Hz, mono",
        ]
        if self.warnings:
            lines.append("Uwagi:")
            lines.extend(f"- {w}" for w in self.warnings)
        return "\n".join(lines)


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    """Wczytuje plik audio, zamienia na mono i resampluje do target_sr.

    librosa.load samo obsługuje mono=True (uśrednienie kanałów) oraz
    resampling przy podanym `sr` - nie trzeba tego robić ręcznie.
    """
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def _slice_on_silence(audio: np.ndarray, top_db: float = 55.0) -> list[np.ndarray]:
    """Dzieli sygnał na fragmenty rozdzielone ciszą.

    `top_db` to próg: fragmenty co najmniej `top_db` decybeli cichsze od
    szczytu sygnału są traktowane jako cisza i wycinane jako granice.
    Domyślnie 55 (wcześniej 40) - mniej agresywne cięcie przy ciągłej mowie
    z mikro-oddechami; librosa domyślnie używa 60.
    """
    if audio.size == 0:
        return []
    intervals = librosa.effects.split(audio, top_db=top_db)
    return [audio[start:end] for start, end in intervals]


def _split_fixed_windows(audio: np.ndarray, sr: int, window_s: float) -> list[np.ndarray]:
    """Dzieli sygnał na kolejne okna o długości window_s (ostatnie może być krótsze)."""
    if audio.size == 0 or window_s <= 0:
        return []
    window = max(1, int(round(window_s * sr)))
    return [audio[i : i + window] for i in range(0, len(audio), window)]


def _apply_duration_filter(
    chunks: list[np.ndarray],
    sr: int,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[np.ndarray], int, int]:
    """Filtr długości: za krótkie odrzuca; za długie tnie na okna max_duration_s.

    Zwraca (kept, rejected_short, resplit_long), gdzie resplit_long to liczba
    zaakceptowanych próbek powstałych z podziału zbyt długich fragmentów.
    """
    kept: list[np.ndarray] = []
    rejected_short = 0
    resplit_long = 0
    for chunk in chunks:
        duration = len(chunk) / sr
        if duration < min_duration_s:
            rejected_short += 1
            continue
        if duration > max_duration_s:
            parts = _split_fixed_windows(chunk, sr, max_duration_s)
            for part in parts:
                part_dur = len(part) / sr
                if part_dur < min_duration_s:
                    rejected_short += 1
                    continue
                kept.append(part)
                resplit_long += 1
            continue
        kept.append(chunk)
    return kept, rejected_short, resplit_long


def _normalize_loudness(chunk: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalizuje głośność fragmentu do target_lufs (EBU R128 / ITU-R BS.1770).

    Bardzo krótkie lub ciche fragmenty mogą dać niezmierzalną głośność
    (pyloudnorm zwraca -inf) - w takim wypadku fragment zostaje bez zmian,
    żeby nie wywołać błędu / nie wzmocnić czystego szumu do pełnej skali.
    """
    meter = pyln.Meter(sr)
    try:
        current_lufs = meter.integrated_loudness(chunk)
    except Exception:  # noqa: BLE001 - zbyt krótki/cichy fragment
        return chunk
    if not np.isfinite(current_lufs):
        return chunk
    normalized = pyln.normalize.loudness(chunk, current_lufs, target_lufs)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    return normalized.astype(np.float32)


def _denoise(chunk: np.ndarray, sr: int) -> np.ndarray:
    """Odszumianie metodą spectral gating (noisereduce), ustawienia ostrożne.

    `prop_decrease` < 1.0 oznacza częściową redukcję szumu - celowo nie
    100%, żeby nie wprowadzać sztucznych, "wodnistych" artefaktów w miejsce
    naturalnego tła (co szkodzi jakości modelu bardziej niż sam szum).
    """
    if chunk.size < sr // 4:
        # Zbyt krótki fragment, żeby sensownie oszacować profil szumu.
        return chunk
    reduced = nr.reduce_noise(y=chunk, sr=sr, prop_decrease=0.75)
    return reduced.astype(np.float32)


def _zip_outputs(paths: list[str], out_dir: Path) -> str:
    """Pakuje wynikowe pliki WAV do jednego archiwum ZIP w tym samym katalogu."""
    zip_path = out_dir / "probki_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=Path(p).name)
    return str(zip_path)


def _read_audio_native(path: str) -> tuple[np.ndarray, int]:
    """Wczytuje audio w oryginalnym SR (bez resamplingu) jako mono float32.

    Używane przy ręcznej edycji (cut/trim/save), zanim użytkownik uruchomi
    pełne przygotowanie datasetu z docelowym SR pod RVC.
    """
    data, sr = sf.read(path, always_2d=True)
    # uśrednienie kanałów → mono (jak w wymaganiu RVC)
    mono = data.mean(axis=1).astype(np.float32)
    return mono, int(sr)


def _write_temp_wav(audio: np.ndarray, sr: int, prefix: str) -> str:
    """Zapisuje tymczasowy WAV PCM_16 i zwraca ścieżkę."""
    out_dir = Path(tempfile.mkdtemp(prefix=prefix))
    out_path = out_dir / "audio.wav"
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return str(out_path)


def preview_segment(
    input_path: str | None,
    start_s: float,
    end_s: float,
) -> tuple[str | None, str]:
    """Wycina tylko [start_s, end_s] do odsłuchu przed usunięciem (cut)."""
    if not input_path:
        return None, "Najpierw wgraj plik audio."
    try:
        audio, sr = _read_audio_native(input_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Błąd wczytywania: {exc}"

    duration = len(audio) / sr
    if start_s < 0 or end_s <= start_s or end_s > duration + 1e-6:
        return None, (
            f"Zakres poza plikiem albo od ≥ do. "
            f"Długość nagrania: {duration:.2f} s."
        )

    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    clip = audio[start_i:end_i]
    if clip.size == 0:
        return None, "Zaznaczony fragment jest pusty."

    path = _write_temp_wav(clip, sr, prefix="rvc_preview_")
    return path, f"Podgląd {start_s:.2f}-{end_s:.2f} s ({len(clip) / sr:.2f} s)."


def cut_segment(
    input_path: str | None,
    start_s: float,
    end_s: float,
) -> tuple[str | None, str]:
    """Usuwa środkowy zakres [start_s, end_s] i skleja to, co zostało.

    Można wywoływać wielokrotnie (kolejne wady w nagraniu). Czasów zawsze
    dotyczy *aktualnego* pliku (po poprzednich cutach/trimach).
    """
    if not input_path:
        return None, "Najpierw wgraj plik audio."
    try:
        audio, sr = _read_audio_native(input_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Błąd wczytywania: {exc}"

    duration = len(audio) / sr
    if start_s < 0 or end_s <= start_s:
        return None, "Podaj poprawny zakres: 0 ≤ od < do."
    if start_s >= duration:
        return None, f"„Od” jest poza nagraniem, które ma {duration:.2f} s."
    end_s = min(end_s, duration)

    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    kept = np.concatenate([audio[:start_i], audio[end_i:]])
    if kept.size == 0:
        return None, "Po wycięciu nic nie zostało - zawęż zakres."

    removed = (end_i - start_i) / sr
    new_dur = len(kept) / sr
    path = _write_temp_wav(kept, sr, prefix="rvc_cut_")
    return path, (
        f"Wycięto {removed:.2f} s, od {start_s:.2f} do {end_s:.2f} s. "
        f"Nowa długość: {new_dur:.2f} s. Możesz wycinać dalej albo zapisać."
    )


def trim_segment(
    input_path: str | None,
    start_s: float,
    end_s: float,
) -> tuple[str | None, str]:
    """Zostawia tylko zakres [start_s, end_s] (przeciwieństwo cut_segment).

    Własna implementacja (nie wbudowany trim Gradio), żeby Trim przechodził
    przez tę samą, prawdziwą funkcję Python co Cut - dzięki temu Undo w UI
    obejmuje oba typy edycji na jednym, wspólnym stosie historii.
    """
    if not input_path:
        return None, "Najpierw wgraj plik audio."
    try:
        audio, sr = _read_audio_native(input_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Błąd wczytywania: {exc}"

    duration = len(audio) / sr
    if start_s < 0 or end_s <= start_s:
        return None, "Podaj poprawny zakres: 0 ≤ od < do."
    end_s = min(end_s, duration)

    start_i = int(start_s * sr)
    end_i = int(end_s * sr)
    kept = audio[start_i:end_i]
    if kept.size == 0:
        return None, "Zaznaczony fragment jest pusty."

    new_dur = len(kept) / sr
    path = _write_temp_wav(kept, sr, prefix="rvc_trim_")
    return path, (
        f"Trim: zostawiono {start_s:.2f}-{end_s:.2f} s. "
        f"Nowa długość: {new_dur:.2f} s."
    )


def save_edited_wav(input_path: str | None) -> tuple[str | None, str]:
    """Kopiuje bieżące (po trim/cut) audio do pliku WAV do pobrania."""
    if not input_path:
        return None, "Brak audio do zapisu - wgraj i ewentualnie przytnij/wytnij."
    try:
        audio, sr = _read_audio_native(input_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"Błąd wczytywania: {exc}"

    out_dir = Path(tempfile.mkdtemp(prefix="rvc_save_"))
    out_path = out_dir / "edytowane_audio.wav"
    sf.write(out_path, audio, sr, subtype="PCM_16")
    return str(out_path), (
        f"Gotowe do pobrania: edytowane_audio.wav "
        f"({len(audio) / sr:.2f} s, {sr} Hz, mono)."
    )


def prepare_audio(
    input_path: str | None,
    target_sr: int,
    do_slice: bool,
    min_duration_s: float,
    max_duration_s: float,
    do_normalize: bool,
    do_denoise: bool,
) -> tuple[list[str] | None, str | None, str]:
    """Główna funkcja zakładki 1: pełny łańcuch obróbki jednego pliku wejściowego.

    Zwraca (lista_ścieżek_wynikowych_lub_None, ścieżka_ZIP_lub_None, tekst_podsumowania_PL).
    """
    if not input_path:
        return None, None, "Najpierw wgraj plik audio."

    try:
        audio = _load_mono(input_path, target_sr)
    except Exception as exc:  # noqa: BLE001 - komunikat trafia do UI
        return None, None, (
            "Błąd wczytywania pliku: "
            f"{exc}\nJeśli to MP3, spróbuj przekonwertować na WAV."
        )

    if audio.size == 0:
        return None, None, "Plik jest pusty albo nie zawiera dźwięku."

    chunks = _slice_on_silence(audio) if do_slice else [audio]
    if not chunks:
        return None, None, "Nie wykryto żadnego fragmentu z dźwiękiem - nagranie to sama cisza?"

    kept, rejected_short, resplit_long = _apply_duration_filter(
        chunks, target_sr, min_duration_s, max_duration_s
    )
    if not kept:
        return None, None, (
            "Wszystkie fragmenty odrzucone przez filtr długości.\n"
            f"Za krótkie: {rejected_short}.\n"
            "Zmniejsz suwak „Min. długość próbki” albo wyłącz "
            "cięcie po ciszy - wtedy długi plik zostanie pocięty na okna max."
        )

    processed: list[np.ndarray] = []
    for chunk in kept:
        c = chunk
        if do_denoise:
            # Odszumianie MUSI być przed normalizacją: spectral gating
            # (noisereduce) zauważalnie zmienia energię sygnału (w testach
            # nawet o ~12 LU), więc normalizacja na końcu gwarantuje
            # docelową głośność niezależnie od tego, ile energii ucięło
            # odszumianie.
            c = _denoise(c, target_sr)
        if do_normalize:
            c = _normalize_loudness(c, target_sr, TARGET_LOUDNESS_LUFS)
        processed.append(c)

    out_dir = Path(tempfile.mkdtemp(prefix="rvc_dataset_"))
    out_paths: list[str] = []
    total_duration = 0.0
    for i, c in enumerate(processed, start=1):
        out_path = out_dir / f"probka_{i:03d}.wav"
        sf.write(out_path, c, target_sr, subtype="PCM_16")
        out_paths.append(str(out_path))
        total_duration += len(c) / target_sr

    zip_path = _zip_outputs(out_paths, out_dir)

    summary = DatasetSummary(
        accepted=len(processed),
        rejected_short=rejected_short,
        resplit_long=resplit_long,
        total_duration_s=total_duration,
        sample_rate=target_sr,
    )
    return out_paths, zip_path, summary.as_text()
