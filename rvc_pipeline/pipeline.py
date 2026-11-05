"""Pipeline RVC.

Zakładka 2 (trening) i Zakładka 3 (inferencja) - obie realne, przez silnik
**Applio** (MIT), wołany jako osobne procesy przez `rvc_pipeline.applio_engine`
/ `rvc_pipeline.applio_train` (patrz `SETUP_APPLIO.md` po konfigurację
środowiska - ten sam `APPLIO_DIR` / `APPLIO_PYTHON` dla obu zakładek).
"""

from __future__ import annotations

import json
from pathlib import Path

from rvc_pipeline.applio_engine import (
    ApplioConfigError,
    ApplioInferenceError,
    DEFAULT_F0_METHOD,
    DEFAULT_INDEX_RATE,
    EMBEDDER_FROM_MODEL,
    F0_METHODS,
    run_inference,
)
from rvc_pipeline.applio_train import (
    DEFAULT_SAVE_EVERY,
    ApplioTrainingError,
    list_drive_checkpoints,
    prepare_checkpoint_downloads,
    retry_model_save,
    run_training,
    sanitize_model_name,
)
from rvc_pipeline.dataset_info import summarize_samples
from rvc_pipeline.train_monitor import (
    TrainingMonitor,
    active_names,
    get_active,
    register,
    unregister,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "profiles.json"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def list_presets() -> list[str]:
    return list(load_config().get("presets", {}).keys())


def preset_choices() -> list[tuple[str, str]]:
    """``(polska_etykieta, klucz)`` dla listy profili w UI."""
    presets = load_config().get("presets", {})
    return [(p.get("label", key), key) for key, p in presets.items()]


def describe_preset(name: str) -> str:
    presets = load_config().get("presets", {})
    p = presets.get(name)
    if not p:
        return f"Nieznany profil: {name}"
    return (
        f"Profil: {p.get('label', name)}\n"
        f"{p.get('description', '')}\n"
        f"Epoki: {p.get('epochs')}\n"
        f"Rozmiar paczki: {p.get('batch_size')}\n"
        f"Metoda F0: {p.get('f0_method')}"
    )


def effective_train_params(
    mode: str, preset: str, epochs, batch_size, f0_method: str | None
) -> tuple[int, int, str]:
    """``(epoki, batch, f0)`` - z presetu w trybie Prosty, z pól w trybie Zaawansowanym."""
    if mode == "Prosty":
        cfg = load_config().get("presets", {}).get(preset, {})
        return (
            int(cfg.get("epochs", 150)),
            int(cfg.get("batch_size", 4)),
            cfg.get("f0_method", DEFAULT_F0_METHOD),
        )
    return int(epochs or 1), int(batch_size or 1), f0_method or DEFAULT_F0_METHOD


def run_train_ui(
    model_name: str,
    sample_files: list | None,
    mode: str,
    preset: str,
    sample_rate: int,
    epochs: int,
    batch_size: int,
    f0_method: str,
    embedder_model: str,
    save_every: int = DEFAULT_SAVE_EVERY,
    auto_backup: bool = False,
    resume: bool = False,
    drive_g_path: str | None = None,
    resume_g_file: str | None = None,
    resume_d_file: str | None = None,
):
    """Wywoływane z przycisku „Rozpocznij trening” w Zakładce 2 (generator Gradio).

    Tryb **Prosty** czerpie epoki/batch/f0 z presetu (`configs/profiles.json`);
    **Zaawansowany** używa wartości z UI. Zwraca kolejne
    ``(zip|None, log, html_postępu, html_wykresu_strat|None, (G, D)|None)``.
    ``None`` przy wykresie = bez zmian od poprzedniego odświeżenia. Ostatni
    element to pliki G/D do pobrania - tylko w ostatnim yieldzie po treningu.

    Źródło checkpointu przy wznowieniu: wgrane pliki G+D, inaczej para
    wybrana z Drive (``drive_g_path``), inaczej ``logs/`` z tej sesji.
    """
    files = [f for f in (sample_files or []) if f]
    if not files and not resume:
        yield None, "Wgraj co najmniej jedną próbkę albo ZIP z zakładki 1.", "", None, None
        return
    try:
        name = sanitize_model_name(model_name)
    except ApplioTrainingError as exc:
        yield None, str(exc), "", None, None
        return

    eff_epochs, eff_batch, eff_f0 = effective_train_params(
        mode, preset, epochs, batch_size, f0_method
    )

    g_path = d_path = None
    source = "checkpoint z tej sesji"
    if resume and (resume_g_file or resume_d_file):
        g_path, d_path = resume_g_file, resume_d_file
        source = "wgrane pliki G/D"
    elif resume and drive_g_path:
        g = Path(drive_g_path)
        g_path, d_path = str(g), str(g.with_name("D_" + g.name[2:]))
        source = f"Drive: {g.name} + D_{g.name[2:]}"

    monitor = TrainingMonitor(name)
    try:
        register(monitor)
    except RuntimeError as exc:
        yield None, f"{exc} Zatrzymaj go albo poczekaj na koniec.", "", None, None
        return

    header = (
        f"{'Wznowienie' if resume else 'Start'} treningu „{name}” - "
        f"epoki łącznie: {eff_epochs}, batch: {eff_batch}, f0: {eff_f0}, "
        f"model cech: {embedder_model}, SR: {sample_rate}, "
        f"checkpoint co {int(save_every)} epok"
        + (f", źródło: {source}" if resume else "")
        + "\n"
    )
    yield None, header, monitor.progress_html(), monitor.loss_chart_html(), None

    zip_path: str | None = None
    last_log = header
    loss_version = monitor.loss_version
    steps = run_training(
        model_name=name,
        sample_paths=files,
        sample_rate=int(sample_rate),
        epochs=eff_epochs,
        batch_size=eff_batch,
        f0_method=eff_f0,
        embedder_model=embedder_model,
        save_every=int(save_every),
        resume=bool(resume),
        resume_g_path=g_path,
        resume_d_path=d_path,
        auto_backup=bool(auto_backup),
        monitor=monitor,
    )
    try:
        for _stage, log, maybe_zip in steps:
            if maybe_zip:
                zip_path = maybe_zip
            chart = None
            if monitor.loss_version != loss_version:
                loss_version = monitor.loss_version
                chart = monitor.loss_chart_html()
            last_log = log
            yield zip_path, log, monitor.progress_html(), chart, None
    except ApplioConfigError as exc:
        last_log = f"Błąd konfiguracji środowiska Applio:\n{exc}"
        yield None, last_log, monitor.progress_html(), None, None
    except ApplioTrainingError as exc:
        last_log = f"Błąd treningu:\n{exc}"
        yield None, last_log, monitor.progress_html(), None, None
    finally:
        # Zamknięty generator (rozłączenie, anulowanie) - _stream_step w swoim
        # finally zabija proces Applio, więc trening nie zostaje „w tle” bez UI.
        monitor.stop_requested = True
        steps.close()
        unregister(monitor)

    g_file, d_file = prepare_checkpoint_downloads(name, monitor.last_ckpt_epoch)
    if g_file or d_file:
        yield zip_path, last_log, monitor.progress_html(), None, (g_file, d_file)


def stop_training_ui(model_name: str) -> str:
    try:
        name = sanitize_model_name(model_name)
    except ApplioTrainingError:
        name = ""
    monitor = get_active(name) if name else None
    if monitor is None:
        running = active_names()
        if len(running) == 1:
            monitor = get_active(running[0])
    if monitor is None:
        return "Nie trwa żaden trening."
    return monitor.request_stop()


def drive_checkpoint_choices(model_name: str) -> tuple[list[tuple[str, str]], str]:
    """Opcje dropdownu „Checkpoint z Google Drive”: ``(etykieta, ścieżka_G)``."""
    pairs, message = list_drive_checkpoints(model_name)
    return [(p.label, str(p.g_path)) for p in pairs], message


def summarize_samples_ui(sample_files, mode, preset, epochs, batch_size) -> str:
    eff_epochs, eff_batch, _ = effective_train_params(mode, preset, epochs, batch_size, None)
    return summarize_samples(sample_files, eff_epochs, eff_batch)


def save_model_drive_ui(zip_path: str | None, model_name: str) -> str:
    return retry_model_save(zip_path, model_name)


def run_infer_ui(
    model_file: str | None,
    index_file: str | None,
    source_audio: str | None,
    pitch: int,
    f0_method: str = DEFAULT_F0_METHOD,
    index_rate: float = DEFAULT_INDEX_RATE,
    embedder_model: str = EMBEDDER_FROM_MODEL,
):
    """Wywoływane z przycisku „Konwertuj” w Zakładce 3 (generator Gradio).

    Najpierw czyści wynik (żeby nie wisiał poprzedni WAV), potem uruchamia
    Applio. Zwraca ``(ścieżka_wav|None, ścieżka_do_pobrania|None, log)``.
    Widoczność odtwarzacza steruje warstwa UI w ``app/main.py``.
    """
    clear_msg = (
        "Trwa konwersja…\n"
        "Odtwarzacz wyniku jest ukryty do czasu zakończenia - nie będzie pustego 00:00.\n"
        "Po zakończeniu pojawi się odsłuch oraz „Pobierz wynik WAV”."
    )
    yield None, None, clear_msg

    if not model_file:
        yield None, None, "Wgraj model: ZIP z .pth i .index albo sam plik .pth."
        return
    if not source_audio:
        yield None, None, "Wgraj nagranie, na które ma zostać nałożona barwa modelu."
        return

    try:
        output_path, log = run_inference(
            model_path=model_file,
            source_audio=source_audio,
            pitch=int(pitch),
            index_path=index_file,
            f0_method=f0_method or DEFAULT_F0_METHOD,
            index_rate=index_rate,
            embedder_model=embedder_model or EMBEDDER_FROM_MODEL,
        )
    except ApplioConfigError as exc:
        yield None, None, f"Błąd konfiguracji środowiska Applio:\n{exc}"
        return
    except ApplioInferenceError as exc:
        yield None, None, f"Błąd konwersji:\n{exc}"
        return

    done = (
        f"Gotowe. Wynik zapisany - możesz odsłuchać lub pobrać WAV.\n"
        f"Plik: {output_path}\n\n"
        f"{log}"
    )
    yield output_path, output_path, done


def on_infer_input_change(path: str | None) -> tuple[None, None, str]:
    """Czyści wynik przy zmianie modelu / źródła - bez starego audio w playerze."""
    if not path:
        return None, None, "Wgraj plik i poczekaj, aż wysyłanie się skończy - w polu widać postęp."
    name = Path(path).name
    return (
        None,
        None,
        f"Wgrano: {name}\n"
        "Poprzedni wynik wyczyszczony. Możesz nacisnąć Konwertuj.",
    )