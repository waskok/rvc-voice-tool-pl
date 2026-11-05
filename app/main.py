"""Polski interfejs Gradio narzędzia RVC - 3 zakładki (MVP A1).

Zakładki:
  1. Przygotowanie nagrań (edycja + dataset)
  2. Trening modelu
  3. Zamiana głosu (konwersja)

Uruchomienie lokalne (warsztat):
    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python -m app.main

W Colabie notebook ustawia APP_SHARE=1 i odpala to samo UI.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import gradio as gr

from audio_tools.process import (
    DEFAULT_SAMPLE_RATE,
    TARGET_SAMPLE_RATES,
    cut_segment,
    prepare_audio,
    trim_segment,
)
from rvc_pipeline.pipeline import (
    describe_preset,
    list_presets,
    preset_choices,
    on_infer_input_change,
    run_infer_ui,
    run_train_ui,
    drive_checkpoint_choices,
    save_model_drive_ui,
    stop_training_ui,
    summarize_samples_ui,
)
from app.help_texts import BUTTON_HELP, FIELD_HELP
from rvc_pipeline import save_log
from rvc_pipeline.train_monitor import AUTO_BACKUP_KEEP, loss_chart_svg
from rvc_pipeline.applio_engine import (
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_F0_METHOD,
    DEFAULT_INDEX_RATE,
    EMBEDDER_FROM_MODEL,
    EMBEDDER_MODELS,
    F0_METHODS,
)
from rvc_pipeline.applio_train import (
    DEFAULT_SAVE_EVERY,
    SAVE_EVERY_MAX,
)

_EDITOR_JS = (Path(__file__).parent / "audio_editor.js").read_text(encoding="utf-8")
_THEME_JS = (Path(__file__).parent / "theme_toggle.js").read_text(encoding="utf-8")
_HELP_JS = (
    (Path(__file__).parent / "help_tooltips.js")
    .read_text(encoding="utf-8")
    .replace("__RVC_FIELD_HELP__", json.dumps(FIELD_HELP))
    .replace("__RVC_BUTTON_HELP__", json.dumps(BUTTON_HELP))
)

MODE_SIMPLE = "Prosty"
MODE_ADVANCED = "Zaawansowany"

# Gradio wybiera język wbudowanych napisów („Upuść plik tutaj”, „Wyczyść”…)
# z navigator.language. Ten skrypt z <head> wymusza "pl"; działa razem
# z launch(i18n=...), bo dopiero niepuste tłumaczenia każą Gradio ponownie
# odczytać język po wstrzyknięciu <head>.
_LOCALE_HEAD = """<script>
(function () {
  var langs = ["pl", "pl-PL"];
  var defs = {
    language: { get: function () { return langs[0]; }, configurable: true },
    languages: { get: function () { return langs; }, configurable: true }
  };
  try { Object.defineProperties(Navigator.prototype, defs); } catch (e) {}
  try { Object.defineProperties(navigator, defs); } catch (e) {}
  document.documentElement.lang = "pl";
})();
</script>"""

_GRADIO_I18N = gr.I18n(pl={"rvc.lang": "pl"})

_HELP_CSS = """
.rvc-help {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  margin-left: 6px;
  border-radius: 50%;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  color: var(--body-text-color-subdued);
  font-size: 11px;
  font-weight: 700;
  line-height: 1;
  vertical-align: middle;
  cursor: help;
  pointer-events: auto;
  user-select: none;
  flex: 0 0 auto;
}
.rvc-help:hover,
.rvc-help:focus-visible {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: white;
  outline: none;
}
.rvc-tip {
  position: fixed;
  z-index: 10000;
  max-width: min(340px, calc(100vw - 16px));
  padding: 10px 12px;
  border-radius: 8px;
  background: #1e293b;
  color: #f8fafc;
  font-size: 0.82rem;
  font-weight: 400;
  line-height: 1.45;
  white-space: pre-line;
  box-shadow: 0 6px 24px rgba(15, 23, 42, 0.25);
  pointer-events: none;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.12s ease;
}
.rvc-tip.is-on {
  opacity: 1;
  visibility: visible;
}
"""

_HIDE_TRIGGER_CSS = """
#rvc_cut_btn, #rvc_trim_btn, #rvc_undo_btn, #rvc_save_btn,
#rvc_cut_start, #rvc_cut_end {
  position: absolute !important;
  left: -9999px !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
}
"""

# Pasek pod odtwarzaczem wyniku, dopóki fala się nie zdekoduje
# (gradio.live: player 0:00 wygląda jak zepsuty). Nie zasłania playera.
_AUDIO_LOADING_CSS = """
#rvc_out_audio .rvc-audio-loading {
  display: none;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--background-fill-secondary);
  border: 1px dashed var(--neutral-400);
  border-radius: 8px;
  font-size: 0.9rem;
  color: var(--body-text-color);
  white-space: pre-line;
  line-height: 1.4;
}
#rvc_out_audio .rvc-audio-loading.is-on {
  display: block;
}
"""

_LAYOUT_CSS = """
.gradio-container {
  max-width: none !important;
  margin: 0 !important;
}
.gradio-container > main.app,
.gradio-container .app.fillable {
  padding-left: clamp(10px, 2vw, 28px) !important;
  padding-right: clamp(10px, 2vw, 28px) !important;
}
footer { display: none !important; }

.rvc-header { padding: 8px 150px 4px 2px; }
#rvc-theme-btn {
  position: fixed;
  top: 10px;
  right: 12px;
  z-index: 10001;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid var(--border-color-primary, #cbd5e1);
  background: var(--block-background-fill, #fff);
  color: var(--body-text-color, #111);
  font: 650 0.9rem/1 ui-sans-serif, system-ui, sans-serif;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.18);
}
#rvc-theme-btn:hover { border-color: var(--color-accent, #2563eb); }
#rvc-theme-btn:focus-visible {
  outline: 2px solid var(--color-accent, #2563eb);
  outline-offset: 2px;
}
.rvc-header h1 {
  margin: 0 0 4px;
  font-size: 1.55rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}
.rvc-header p {
  margin: 0;
  color: var(--body-text-color-subdued);
  font-size: 0.95rem;
}
.rvc-flow {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  margin-top: 10px;
  font-size: 0.92rem;
  color: var(--body-text-color-subdued);
}
.rvc-flow b { color: var(--body-text-color); font-weight: 600; }

.rvc-card {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  background: var(--block-background-fill) !important;
  gap: 12px !important;
}

.rvc-step {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin: 2px 0 0;
}
.rvc-step-num {
  flex: 0 0 auto;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color);
  font-size: 0.8rem;
  font-weight: 650;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-top: 1px;
}
.rvc-step-title { font-weight: 600; font-size: 1rem; line-height: 1.5; }
.rvc-step-hint {
  display: block;
  font-weight: 400;
  font-size: 0.9rem;
  color: var(--body-text-color-subdued);
  line-height: 1.4;
}

.rvc-intro { font-size: 0.92rem; color: var(--body-text-color-subdued); }
.rvc-intro p { margin: 0; }

.rvc-note {
  font-size: 0.9rem;
  color: var(--body-text-color-subdued);
  border-left: 3px solid var(--neutral-400);
  padding: 2px 0 2px 10px;
}
.rvc-note p { margin: 0; }

.rvc-next {
  border-left: 3px solid var(--color-accent);
  padding: 6px 0 6px 12px;
  font-size: 0.95rem;
}
.rvc-next p { margin: 0; }

.rvc-progress {
  border: 1px solid var(--border-color-primary);
  border-radius: 10px;
  padding: 10px 12px;
  font-size: 0.9rem;
}
.rvc-progress-title {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
}
.rvc-progress-pct { font-weight: 650; }
.rvc-progress-bar {
  height: 8px;
  border-radius: 999px;
  background: var(--border-color-primary);
  overflow: hidden;
}
.rvc-progress-bar > div {
  height: 100%;
  background: var(--color-accent);
  transition: width 0.4s ease;
}
.rvc-progress-details {
  margin: 8px 0 0;
  padding-left: 18px;
  color: var(--body-text-color-subdued);
  font-size: 0.88rem;
}

.rvc-footer {
  margin-top: 8px;
  font-size: 0.85rem;
  color: var(--body-text-color-subdued);
  text-align: center;
}
.rvc-footer p { margin: 0; }

.rvc-block-title {
  display: inline-flex;
  align-items: center;
  font-size: var(--block-label-text-size);
  font-weight: var(--block-label-text-weight);
  color: var(--block-label-text-color);
  margin: 2px 0 -4px;
}

/* Logi: gr.HTML w kontenerze o stałej wysokości. Kontener (.html-container)
   nie jest podmieniany przy odświeżeniu - zmienia się tylko <pre> w środku,
   więc pozycja suwaka zostaje, a strona nie podskakuje. */
#rvc_train_log .html-container,
#rvc_infer_log .html-container,
#rvc_save_log .html-container {
  height: 26rem;
  overflow-y: auto !important;
  overflow-anchor: none;
  padding: 0 !important;
  opacity: 1 !important;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--block-radius);
  background: var(--background-fill-secondary);
}
#rvc_infer_log .html-container { height: 16rem; }
#rvc_save_log .html-container { height: 11rem; }
.rvc-log {
  margin: 0 !important;
  padding: 10px 12px !important;
  background: transparent !important;
  border: none !important;
  font-family: var(--font-mono);
  font-size: 0.84rem;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--body-text-color);
}
#rvc_train_progress { min-height: 9rem; }
#rvc_train_progress .html-container,
#rvc_loss_chart .html-container { opacity: 1 !important; }

.rvc-chart { width: 100%; }
.rvc-chart-panel + .rvc-chart-panel { margin-top: 14px; }
.rvc-chart-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 2px 12px;
  margin-bottom: 2px;
  font-size: 0.9rem;
  color: var(--body-text-color);
}
.rvc-chart-head > span:last-child { color: var(--body-text-color-subdued); }
.rvc-chart-head b { color: var(--body-text-color); font-weight: 650; }
.rvc-chart-head i {
  display: inline-block;
  width: 14px;
  height: 4px;
  margin-right: 7px;
  vertical-align: middle;
  border-radius: 2px;
}
.rvc-chart-svg { width: 100%; height: auto; display: block; }
.rvc-chart-grid { stroke: var(--border-color-primary); stroke-width: 1; }
.rvc-chart-axis { stroke: var(--body-text-color-subdued); stroke-width: 1.2; }
.rvc-chart-tick { fill: var(--body-text-color-subdued); font-size: 12px; }
#rvc_dl_row { gap: 10px; }
#rvc_dl_row > * { min-width: 160px; }
.rvc-chart-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin-top: 4px;
  font-size: 0.88rem;
  color: var(--body-text-color-subdued);
}
.rvc-chart-legend i {
  display: inline-block;
  width: 12px;
  height: 3px;
  margin-right: 6px;
  vertical-align: middle;
  border-radius: 2px;
}
.rvc-chart-empty {
  min-height: 6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 12px;
  border: 1px dashed var(--border-color-primary);
  border-radius: 8px;
  font-size: 0.9rem;
  color: var(--body-text-color-subdued);
}
"""

# Scroll logu (gr.HTML): kontener .html-container zostaje między odświeżeniami,
# więc przeglądarka sama trzyma pozycję suwaka. JS tylko dociąga na dół, gdy
# użytkownik był na dole (stan liczony z jego własnych zdarzeń scroll). Bez
# MutationObservera i bez ruszania scrolla całej strony.
_LOG_SCROLL_JS = """
() => {
  if (window.__rvcLogScrollInit) return;
  window.__rvcLogScrollInit = true;

  const NEAR_PX = 24;
  const IDS = ["rvc_train_log", "rvc_infer_log", "rvc_save_log"];

  const bind = (box) => {
    if (box.__rvcLog) return box.__rvcLog;
    const st = { stick: true };
    box.addEventListener(
      "scroll",
      () => {
        st.stick = box.scrollHeight - box.scrollTop - box.clientHeight < NEAR_PX;
      },
      { passive: true }
    );
    box.__rvcLog = st;
    return st;
  };

  const tick = () => {
    for (const id of IDS) {
      const box = document.querySelector(`#${id} .html-container`);
      if (!box) continue;
      const st = bind(box);
      const gap = box.scrollHeight - box.scrollTop - box.clientHeight;
      if (st.stick && gap > 1) box.scrollTop = box.scrollHeight;
    }
  };

  tick();
  setInterval(tick, 250);
}
"""

_AUDIO_LOADING_JS = """
() => {
  // Jednorazowa inicjalizacja - Gradio potrafi reinjectować JS przy reconnect.
  if (window.__rvcAudioLoadingInit) return;
  window.__rvcAudioLoadingInit = true;

  const TEXT = "Wczytywanie odtwarzacza…\\nPlik jest już gotowy - możesz go pobrać poniżej przyciskiem \\u201ePobierz wynik WAV\\u201d, nie trzeba czekać.";
  const SLOW_TEXT = "Odtwarzacz wczytuje się dłużej niż zwykle - przy długim nagraniu przez link gradio.live to normalne.\\nPlik możesz już pobrać poniżej przyciskiem \\u201ePobierz wynik WAV\\u201d.";
  const SLOW_MS = 45000;
  const state = { key: null, since: 0 };

  // Gradio 5 z falą odtwarza przez WaveSurfer: <audio class="standard-player">
  // w DOM jest ukryty i nigdy nie dostaje src, więc jego duration zostaje NaN.
  // Po zdekodowaniu pliku WaveSurfer wpisuje długość do <time id="duration">.
  const isReady = (root) => {
    const dur = root.querySelector("#duration");
    const t = dur ? dur.textContent.trim() : "";
    if (t && t !== "0:00") return true;
    for (const a of root.querySelectorAll("audio")) {
      if (Number.isFinite(a.duration) && a.duration > 0) return true;
    }
    return false;
  };

  // WAŻNE: NIE używać MutationObserver na document.body.
  // tick() zmienia textContent/classList → mutation → tick → pętla → RAM GB+.
  // Tylko polling + zapis DOM wyłącznie gdy stan się zmienia.
  const tick = () => {
    const root = document.getElementById("rvc_out_audio");
    if (!root) return;
    const hasValue = root.querySelector("audio, #waveform") !== null;
    const link = root.querySelector("a[href]");
    const key = hasValue ? (link ? link.href : "wynik") : null;
    if (key !== state.key) {
      state.key = key;
      state.since = Date.now();
    }

    let badge = root.querySelector(".rvc-audio-loading");
    if (!badge) {
      if (key === null) return;
      badge = document.createElement("div");
      badge.className = "rvc-audio-loading";
      root.appendChild(badge);
    }
    const nextOn = key !== null && !isReady(root);
    const nextText = Date.now() - state.since > SLOW_MS ? SLOW_TEXT : TEXT;
    if (badge.textContent !== nextText) badge.textContent = nextText;
    if (badge.classList.contains("is-on") !== nextOn) {
      badge.classList.toggle("is-on", nextOn);
    }
  };

  tick();
  setInterval(tick, 500);
}
"""

_NEXT_AFTER_PREP = (
    "**Dalej:** pobierz ZIP powyżej i przejdź do zakładki "
    "**2 · Wytrenuj model** - tam wgrasz go jako próbki treningowe."
)
_NEXT_AFTER_TRAIN = (
    "**Dalej:** pobierz ZIP modelu w kroku 5 poniżej i przejdź do zakładki "
    "**3 · Zamień głos** - tam wgrasz go jako model."
)
_NEXT_AFTER_INFER = (
    "**Gotowe.** Odsłuchaj wynik albo pobierz plik WAV. "
    "Możesz zmienić wysokość głosu lub nagranie i kliknąć **Zamień głos** ponownie."
)


def _build_theme() -> gr.themes.Base:
    system_font = [
        "ui-sans-serif",
        "system-ui",
        "-apple-system",
        "Segoe UI",
        "Roboto",
        "Helvetica Neue",
        "Arial",
        "sans-serif",
    ]
    mono_font = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_md,
        # Skala o stopień większa od domyślnej - czytelna z rzutnika.
        text_size=gr.themes.Size(
            xxs="10px", xs="11px", sm="13px", md="15px", lg="18px", xl="22px", xxl="26px"
        ),
        font=system_font,
        font_mono=mono_font,
    ).set(
        # Jasny motyw: białe tło i prawie czarny tekst, czytelne z rzutnika.
        body_background_fill="white",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_950",
        body_text_color_dark="*neutral_50",
        body_text_color_subdued="*neutral_800",
        body_text_color_subdued_dark="*neutral_300",
        border_color_primary="*neutral_300",
        border_color_primary_dark="*neutral_600",
        block_background_fill="white",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_shadow="none",
        block_label_text_color="*neutral_950",
        block_label_text_color_dark="*neutral_200",
        block_label_text_weight="600",
        block_title_text_color="*neutral_950",
        block_title_text_color_dark="*neutral_100",
        block_title_text_weight="600",
        block_info_text_color="#333333",
        block_info_text_color_dark="*neutral_300",
        input_background_fill="white",
        input_background_fill_dark="*neutral_800",
        input_border_color="*neutral_400",
        input_border_color_dark="*neutral_500",
        input_border_width="1px",
        input_placeholder_color="*neutral_500",
        input_placeholder_color_dark="*neutral_400",
        checkbox_border_color="*neutral_500",
        checkbox_border_color_dark="*neutral_400",
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200",
        button_secondary_background_fill_dark="*neutral_700",
        button_secondary_background_fill_hover_dark="*neutral_600",
        button_secondary_border_color="*neutral_400",
        button_secondary_border_color_dark="*neutral_500",
        button_secondary_text_color="*neutral_900",
        button_secondary_text_color_dark="white",
        button_primary_background_fill="*primary_800",
        button_primary_background_fill_hover="*primary_700",
        button_primary_background_fill_dark="*primary_600",
        button_primary_background_fill_hover_dark="*primary_500",
        button_primary_text_color="white",
        button_primary_text_color_dark="white",
        button_primary_border_color="*primary_800",
        button_primary_border_color_dark="*primary_600",
        color_accent="*primary_600",
        color_accent_soft="*primary_50",
        color_accent_soft_dark="*neutral_800",
        slider_color="*primary_700",
        slider_color_dark="*primary_500",
        checkbox_background_color_selected="*primary_700",
        checkbox_background_color_selected_dark="*primary_500",
        checkbox_border_color_selected="*primary_700",
        checkbox_border_color_selected_dark="*primary_500",
    )


def _step(num: int, title: str, hint: str = "") -> gr.HTML:
    hint_html = f'<span class="rvc-step-hint">{hint}</span>' if hint else ""
    return gr.HTML(
        '<div class="rvc-step">'
        f'<span class="rvc-step-num">{num}</span>'
        f'<span class="rvc-step-title">{title}{hint_html}</span>'
        "</div>"
    )


def _next_hint(show: bool, text: str):
    return gr.update(value=text if show else "", visible=show)


def _prepare_audio_ui(
    path, target_sr, do_slice, min_duration, max_duration, do_normalize, do_denoise
):
    files, zip_path, info = prepare_audio(
        path, target_sr, do_slice, min_duration, max_duration, do_normalize, do_denoise
    )
    return files, zip_path, info, _next_hint(bool(zip_path), _NEXT_AFTER_PREP)


def _log_html(text: str | None) -> str:
    return f'<pre class="rvc-log">{html.escape(text or "")}</pre>'


def _block_title(text: str, elem_id: str) -> gr.HTML:
    return gr.HTML(f'<span class="rvc-block-title">{text}</span>', elem_id=elem_id)


def _save_log_html() -> str:
    return _log_html(save_log.text())


def _download_update(path: str | None):
    return gr.update(value=path, interactive=bool(path))


def _train_ui_stream(model_name, *args):
    first = True
    last_zip = None
    log_version = -1
    for zip_path, log, progress, chart, ckpts in run_train_ui(model_name, *args):
        zip_up = gr.update()
        if first or zip_path != last_zip:
            last_zip = zip_path
            zip_up = _download_update(zip_path)
        g_up = d_up = gr.update()
        if first:
            g_up = d_up = _download_update(None)
        if ckpts is not None:
            g_up, d_up = (_download_update(p) for p in ckpts)
        save_up = gr.update()
        if save_log.version() != log_version:
            log_version = save_log.version()
            save_up = _save_log_html()
        first = False
        yield (
            zip_path,
            zip_up,
            _log_html(log),
            _next_hint(bool(zip_path), _NEXT_AFTER_TRAIN),
            progress,
            gr.update() if chart is None else chart,
            g_up,
            d_up,
            save_up,
            model_name,
        )


def _retry_drive_save(zip_path: str | None, trained_name: str, model_name: str) -> str:
    save_model_drive_ui(zip_path, trained_name or model_name or "model")
    return _save_log_html()


def _refresh_drive_ckpts(resume: bool, model_name: str):
    if not resume:
        return gr.update(), gr.update()
    choices, message = drive_checkpoint_choices(model_name or "")
    return (
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        message,
    )


def _on_resume_change(resume: bool):
    return gr.update(visible=bool(resume))


def _infer_ui_stream(
    model, index_file, source, pitch, f0_method, index_rate, embedder_model
):
    """Owija ``run_infer_ui``: odtwarzacz ukryty w trakcie, widoczny dopiero z wynikiem."""
    for wav, download, log in run_infer_ui(
        model,
        index_file,
        source,
        pitch,
        f0_method,
        index_rate,
        embedder_model,
    ):
        hint = _next_hint(bool(wav), _NEXT_AFTER_INFER)
        if wav:
            yield gr.update(value=wav, visible=True), download, _log_html(log), hint
        else:
            yield gr.update(value=None, visible=False), download, _log_html(log), hint


def _on_infer_input_ui(path: str | None):
    """Czyści wynik i chowa odtwarzacz przy zmianie wejścia."""
    _a, _d, log = on_infer_input_change(path)
    return (
        gr.update(value=None, visible=False),
        None,
        _log_html(log),
        _next_hint(False, ""),
    )


def _on_preset_change(name: str) -> str:
    return describe_preset(name)


def _on_mode_change(mode: str):
    """Prosty: widać profil (preset). Zaawansowany: widać pola parametrów."""
    simple = mode == MODE_SIMPLE
    return (
        gr.update(visible=simple),
        gr.update(visible=simple),
        gr.update(visible=not simple),
    )


def _on_audio_upload(path: str | None):
    """Nowe wgranie - świeża historia cofania."""
    return path, []


def _on_audio_change(path: str | None):
    return path


def _apply_edit(op_fn, path: str | None, start_s: float, end_s: float, undo_stack: list):
    """Wspólna ścieżka dla Trim i Cut - dzięki temu oba trafiają na jeden
    stos historii i Undo (Ctrl+Z) działa tak samo dla obu akcji."""
    if not path:
        return path, path, undo_stack
    if start_s is None or end_s is None or float(start_s) < 0 or float(end_s) < 0:
        return path, path, undo_stack
    new_path, _msg = op_fn(path, float(start_s), float(end_s))
    if new_path is None:
        return path, path, undo_stack
    stack = list(undo_stack or [])
    stack.append(path)
    return new_path, new_path, stack


def _do_cut(path: str | None, start_s: float, end_s: float, undo_stack: list):
    return _apply_edit(cut_segment, path, start_s, end_s, undo_stack)


def _do_trim(path: str | None, start_s: float, end_s: float, undo_stack: list):
    return _apply_edit(trim_segment, path, start_s, end_s, undo_stack)


def _do_undo(undo_stack: list):
    stack = list(undo_stack or [])
    if not stack:
        return gr.update(), gr.update(), stack
    prev = stack.pop()
    return prev, prev, stack


def build_ui() -> gr.Blocks:
    presets = list_presets()
    default_preset = presets[0] if presets else None

    combined_js = (
        "() => {\n"
        f"  ({_THEME_JS})();\n"
        f"  ({_EDITOR_JS})();\n"
        f"  ({_AUDIO_LOADING_JS})();\n"
        f"  ({_HELP_JS})();\n"
        f"  ({_LOG_SCROLL_JS})();\n"
        "}\n"
    )

    with gr.Blocks(
        title="RVC Colab PL",
        theme=_build_theme(),
        js=combined_js,
        head=_LOCALE_HEAD,
        fill_width=True,
        css=_HIDE_TRIGGER_CSS
        + _AUDIO_LOADING_CSS
        + _LAYOUT_CSS
        + _HELP_CSS,
    ) as demo:
        gr.HTML(
            '<div class="rvc-header">'
            "<h1>Konwersja głosu RVC</h1>"
            "<p>Przygotuj nagrania, wytrenuj model swojego głosu i nałóż jego "
            "barwę na inne nagranie. Przejdź przez zakładki po kolei.</p>"
            '<div class="rvc-flow">'
            "<span><b>1</b> Przygotuj nagrania</span>"
            "<span>→</span>"
            "<span><b>2</b> Wytrenuj model</span>"
            "<span>→</span>"
            "<span><b>3</b> Zamień głos</span>"
            "</div>"
            "</div>"
        )

        with gr.Tab("1 · Przygotuj nagrania"):
            gr.Markdown(
                "Z surowego nagrania powstaje zestaw krótkich, czystych próbek "
                "do treningu. Jeśli masz już gotowe próbki - przejdź "
                "od razu do zakładki 2.",
                elem_classes="rvc-intro",
            )
            working_audio = gr.State(None)
            undo_stack = gr.State([])

            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes="rvc-card"):
                    _step(
                        1,
                        "Wgraj nagranie",
                        "Opcjonalnie przytnij je: ikona nożyczek → zaznacz fragment "
                        "na fali → Przytnij / Wytnij / Cofnij Ctrl+Z / Zapisz.",
                    )
                    # Widoczne (ale schowane CSS-em) - trzeba, żeby JS mógł wpisać
                    # czas zaznaczenia z fali przed kliknięciem prawdziwego "Wytnij".
                    cut_start = gr.Number(value=-1, elem_id="rvc_cut_start")
                    cut_end = gr.Number(value=-1, elem_id="rvc_cut_end")

                    in_audio = gr.Audio(
                        label="Nagranie źródłowe",
                        type="filepath",
                        editable=True,
                        elem_id="rvc_edit_audio",
                    )
                    cut_btn = gr.Button("cut", elem_id="rvc_cut_btn")
                    trim_btn = gr.Button("trim", elem_id="rvc_trim_btn")
                    undo_btn = gr.Button("undo", elem_id="rvc_undo_btn")
                    save_btn = gr.DownloadButton(
                        "save", elem_id="rvc_save_btn", value=None
                    )

                    _step(
                        2,
                        "Ustaw przetwarzanie",
                        "Domyślne ustawienia pasują do większości nagrań mowy.",
                    )
                    target_sr = gr.Dropdown(
                        choices=TARGET_SAMPLE_RATES,
                        value=DEFAULT_SAMPLE_RATE,
                        label="Częstotliwość próbkowania (Hz)",
                        info="Wybierz tę samą wartość w zakładce 2.",
                        elem_id="rvc_prep_sr",
                    )
                    with gr.Row():
                        do_slice = gr.Checkbox(
                            value=True,
                            label="Tnij po ciszy",
                            info="Wyłączone → nagranie i tak zostanie podzielone na okna o maks. długości.",
                            elem_id="rvc_do_slice",
                        )
                        do_normalize = gr.Checkbox(
                            value=True,
                            label="Wyrównaj głośność",
                            info="Normalizacja EBU R128.",
                            elem_id="rvc_do_normalize",
                        )
                        do_denoise = gr.Checkbox(
                            value=True,
                            label="Odszumiaj",
                            info="Redukcja stałego szumu tła.",
                            elem_id="rvc_do_denoise",
                        )
                    with gr.Accordion(
                        "Długość próbek", open=False, elem_id="rvc_len_acc"
                    ):
                        min_duration = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            step=0.1,
                            value=1.0,
                            label="Min. długość próbki (s)",
                            info="Krótsze fragmenty, np. oddechy i kliknięcia, są odrzucane.",
                            elem_id="rvc_min_dur",
                        )
                        max_duration = gr.Slider(
                            minimum=5.0,
                            maximum=20.0,
                            step=0.5,
                            value=12.0,
                            label="Maks. długość próbki (s)",
                            info="Dłuższe fragmenty są dzielone na okna tej długości.",
                            elem_id="rvc_max_dur",
                        )
                    prep_btn = gr.Button(
                        "Przygotuj próbki", variant="primary", size="lg", elem_id="rvc_prep_btn"
                    )

                with gr.Column(scale=2, elem_classes="rvc-card"):
                    _step(3, "Pobierz wynik", "Pojawi się tu po przetworzeniu.")
                    out_zip = gr.File(label="Gotowe próbki ZIP", elem_id="rvc_out_zip")
                    out_info = gr.Textbox(
                        label="Podsumowanie", lines=8, interactive=False, elem_id="rvc_out_info"
                    )
                    prep_next = gr.Markdown(visible=False, elem_classes="rvc-next")
                    with gr.Accordion(
                        "Pojedyncze próbki WAV", open=False, elem_id="rvc_out_files_acc"
                    ):
                        out_files = gr.File(
                            label="Próbki",
                            file_count="multiple",
                        )

            in_audio.upload(
                _on_audio_upload,
                inputs=in_audio,
                outputs=[working_audio, undo_stack],
            )
            in_audio.change(
                _on_audio_change,
                inputs=in_audio,
                outputs=working_audio,
            )

            # JS (audio_editor.js) wpisuje czas zaznaczenia do cut_start/cut_end
            # PRZED kliknięciem tego przycisku - dalej to zwykły, prawdziwy
            # klik Gradio z prawdziwą funkcją Python (bez fn=None/js tutaj,
            # bo takie połączenie w tej wersji Gradio nic nie wysyła do backendu).
            # Trim i Cut używają WŁASNEJ implementacji (nie wbudowanego trimu
            # Gradio) i wspólnego undo_stack - dzięki temu Undo działa dla obu.
            cut_btn.click(
                _do_cut,
                inputs=[in_audio, cut_start, cut_end, undo_stack],
                outputs=[in_audio, working_audio, undo_stack],
            )
            trim_btn.click(
                _do_trim,
                inputs=[in_audio, cut_start, cut_end, undo_stack],
                outputs=[in_audio, working_audio, undo_stack],
            )
            undo_btn.click(
                _do_undo,
                inputs=undo_stack,
                outputs=[in_audio, working_audio, undo_stack],
            )
            # Save = zwykły DownloadButton; JS tylko klika go z ikony na fali.
            working_audio.change(lambda p: p, inputs=working_audio, outputs=save_btn)

            prep_btn.click(
                _prepare_audio_ui,
                inputs=[
                    working_audio,
                    target_sr,
                    do_slice,
                    min_duration,
                    max_duration,
                    do_normalize,
                    do_denoise,
                ],
                outputs=[out_files, out_zip, out_info, prep_next],
            )

        with gr.Tab("2 · Wytrenuj model"):
            gr.Markdown(
                "Model uczy się barwy głosu z próbek. Trening trwa od kilku minut "
                "do kilku godzin, postęp po prawej "
                "odświeża się na bieżąco.",
                elem_classes="rvc-intro",
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes="rvc-card"):
                    _step(
                        1,
                        "Nazwij model i wgraj próbki",
                        "Można wgrać kilka plików naraz: ZIP-y z zakładki 1 i/lub pojedyncze nagrania.",
                    )
                    model_name = gr.Textbox(
                        label="Nazwa modelu",
                        placeholder="np. moj_glos",
                        elem_id="rvc_model_name",
                    )
                    samples = gr.File(
                        label="Próbki treningowe",
                        file_count="multiple",
                        type="filepath",
                        file_types=[".zip", ".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                        elem_id="rvc_samples",
                    )
                    dataset_summary = gr.Markdown(elem_classes="rvc-note")
                    sample_rate = gr.Dropdown(
                        choices=TARGET_SAMPLE_RATES,
                        value=DEFAULT_SAMPLE_RATE,
                        label="Częstotliwość próbkowania (Hz)",
                        info="Ta sama co w zakładce 1.",
                        elem_id="rvc_train_sr",
                    )

                    _step(
                        2,
                        "Wybierz ustawienia treningu",
                        "Tryb prosty: gotowy profil ustawień. Zaawansowany: parametry ręcznie.",
                    )
                    mode = gr.Radio(
                        choices=[MODE_SIMPLE, MODE_ADVANCED],
                        value=MODE_SIMPLE,
                        label="Tryb",
                        elem_id="rvc_mode",
                    )
                    preset = gr.Dropdown(
                        choices=preset_choices(),
                        value=default_preset,
                        label="Profil ustawień",
                        elem_id="rvc_preset",
                    )
                    preset_info = gr.Textbox(
                        label="Co ustawia ten profil",
                        elem_id="rvc_preset_info",
                        lines=5,
                        interactive=False,
                        value=describe_preset(default_preset) if default_preset else "",
                    )
                    with gr.Group(visible=False) as advanced_box:
                        with gr.Row():
                            epochs = gr.Number(
                                value=200, precision=0, label="Epoki", elem_id="rvc_epochs"
                            )
                            batch_size = gr.Number(
                                value=8,
                                precision=0,
                                label="Rozmiar paczki",
                                elem_id="rvc_batch",
                            )
                        f0_method = gr.Dropdown(
                            choices=list(F0_METHODS),
                            value=DEFAULT_F0_METHOD,
                            label="Metoda odczytu wysokości głosu",
                            info="rmvpe - zalecana.",
                            elem_id="rvc_f0",
                        )
                        embedder_model = gr.Dropdown(
                            choices=list(EMBEDDER_MODELS),
                            value=DEFAULT_EMBEDDER_MODEL,
                            label="Model cech mowy",
                            info="contentvec - domyślny w Applio; inne warianty dla wybranych języków.",
                            elem_id="rvc_embedder",
                        )
                    save_every = gr.Slider(
                        minimum=5,
                        maximum=SAVE_EVERY_MAX,
                        step=5,
                        value=DEFAULT_SAVE_EVERY,
                        label="Zapisuj checkpoint co N epok",
                        elem_id="rvc_save_every",
                        info=(
                            "W trakcie zapisywane są tylko pliki G/D, mały model "
                            "powstaje na końcu. Epoki najlepiej podzielne przez N."
                        ),
                    )
                    auto_backup = gr.Checkbox(
                        value=True,
                        label="Automatycznie kopiuj checkpointy G/D na Dysk Google",
                        elem_id="rvc_auto_backup",
                        info=(
                            "Po każdym zapisie checkpointu → MyDrive/RVC_Colab_Models/<nazwa>/checkpoints/. "
                            f"Zostają {AUTO_BACKUP_KEEP} ostatnie pary. "
                            "Wymaga komórki „Podłącz Google Drive”."
                        ),
                    )

                    _step(
                        3,
                        "Wznowienie treningu",
                        "Kontynuacja przerwanego lub zakończonego treningu od ostatniego checkpointu.",
                    )
                    resume = gr.Checkbox(
                        value=False,
                        label="Wznów trening z checkpointu",
                        info="Epoki = łączny cel, np. checkpoint z epoki 100 i 200 epok → dotrenuje 101-200.",
                        elem_id="rvc_resume",
                    )
                    with gr.Group(visible=False) as resume_box:
                        gr.Markdown(
                            "**Ta sama sesja Colaba:** wystarczy ta sama nazwa modelu - "
                            "nic nie wybieraj, próbki nie są potrzebne.  \n"
                            "**Nowa sesja:** wgraj **te same** próbki co w pierwszym "
                            "treningu i wybierz parę G/D z Dysku Google albo wgraj "
                            "pliki ręcznie. Ustaw tę samą częstotliwość próbkowania "
                            "i rozmiar paczki.",
                            elem_classes="rvc-note",
                        )
                        with gr.Row(equal_height=True):
                            drive_ckpt = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="Para checkpointów G i D z Dysku Google",
                                scale=4,
                                elem_id="rvc_drive_ckpt",
                            )
                            drive_ckpt_refresh = gr.Button(
                                "Odśwież listę", scale=1, elem_id="rvc_ckpt_refresh"
                            )
                        drive_ckpt_info = gr.Markdown(elem_classes="rvc-note")
                        with gr.Accordion(
                            "Albo wgraj pliki G i D ręcznie",
                            open=False,
                            elem_id="rvc_manual_ckpt_acc",
                        ):
                            with gr.Row():
                                resume_g = gr.File(
                                    label="Checkpoint G",
                                    type="filepath",
                                    file_types=[".pth"],
                                    elem_id="rvc_resume_g",
                                )
                                resume_d = gr.File(
                                    label="Checkpoint D",
                                    type="filepath",
                                    file_types=[".pth"],
                                    elem_id="rvc_resume_d",
                                )
                    with gr.Row():
                        train_btn = gr.Button(
                            "Rozpocznij trening",
                            variant="primary",
                            size="lg",
                            scale=3,
                            elem_id="rvc_train_btn",
                        )
                        stop_btn = gr.Button(
                            "Zatrzymaj trening",
                            variant="stop",
                            size="lg",
                            scale=2,
                            elem_id="rvc_stop_btn",
                        )
                    stop_status = gr.Markdown(elem_classes="rvc-note")

                with gr.Column(scale=3, elem_classes="rvc-card"):
                    _step(
                        4,
                        "Postęp treningu",
                        "Nie zamykaj karty w trakcie treningu - zamknięcie przerywa trening.",
                    )
                    train_progress = gr.HTML(elem_id="rvc_train_progress")
                    _block_title("Straty treningu", "rvc_loss_title")
                    loss_chart = gr.HTML(value=loss_chart_svg([]), elem_id="rvc_loss_chart")
                    gr.Markdown(
                        "Niżej = lepiej. Gdy krzywe przestają spadać przez wiele "
                        "epok, dalszy trening niewiele daje.",
                        elem_classes="rvc-note",
                    )
                    train_next = gr.Markdown(visible=False, elem_classes="rvc-next")
                    _block_title("Dziennik treningu", "rvc_train_log_title")
                    train_log = gr.HTML(value=_log_html(""), elem_id="rvc_train_log")

                    _step(
                        5,
                        "Zapisz model",
                        "Jeśli Dysk Google jest podłączony, model zapisuje się automatycznie "
                        "po treningu w MyDrive/RVC_Colab_Models/, a checkpointy G/D w "
                        "MyDrive/RVC_Colab_Models/<nazwa>/checkpoints/.",
                    )
                    train_zip_state = gr.State(None)
                    train_name_state = gr.State("")
                    with gr.Row(elem_id="rvc_dl_row"):
                        zip_dl = gr.DownloadButton(
                            "Pobierz plik ZIP",
                            value=None,
                            interactive=False,
                            variant="primary",
                            elem_id="rvc_zip_dl",
                        )
                        g_dl = gr.DownloadButton(
                            "Pobierz checkpoint G",
                            value=None,
                            interactive=False,
                            elem_id="rvc_g_dl",
                        )
                        d_dl = gr.DownloadButton(
                            "Pobierz checkpoint D",
                            value=None,
                            interactive=False,
                            elem_id="rvc_d_dl",
                        )
                    gr.Markdown(
                        "ZIP to gotowy model do zakładki 3. G i D są potrzebne tylko do "
                        "wznowienia treningu w nowej sesji - zawsze pobieraj oba.",
                        elem_classes="rvc-note",
                    )
                    with gr.Row(equal_height=True):
                        gr.Markdown(
                            "Model nie zapisał się na Dysku? Podłącz Dysk w notatniku, "
                            "wpisz tę samą nazwę modelu i kliknij obok.",
                            elem_classes="rvc-note",
                        )
                        drive_retry_btn = gr.Button(
                            "Zapisz model na Dysku ponownie",
                            elem_id="rvc_drive_model_btn",
                        )
                    _block_title("Status zapisu", "rvc_save_log_title")
                    save_log_box = gr.HTML(value=_save_log_html, elem_id="rvc_save_log")

            mode.change(
                _on_mode_change,
                inputs=mode,
                outputs=[preset, preset_info, advanced_box],
            )
            preset.change(_on_preset_change, inputs=preset, outputs=preset_info)

            summary_inputs = [samples, mode, preset, epochs, batch_size]
            for trigger in (samples.change, mode.change, preset.change, epochs.blur, batch_size.blur):
                trigger(summarize_samples_ui, inputs=summary_inputs, outputs=dataset_summary)

            train_btn.click(
                _train_ui_stream,
                inputs=[
                    model_name,
                    samples,
                    mode,
                    preset,
                    sample_rate,
                    epochs,
                    batch_size,
                    f0_method,
                    embedder_model,
                    save_every,
                    auto_backup,
                    resume,
                    drive_ckpt,
                    resume_g,
                    resume_d,
                ],
                outputs=[
                    train_zip_state,
                    zip_dl,
                    train_log,
                    train_next,
                    train_progress,
                    loss_chart,
                    g_dl,
                    d_dl,
                    save_log_box,
                    train_name_state,
                ],
                concurrency_limit=1,
                concurrency_id="rvc_train",
                scroll_to_output=False,
                show_progress="hidden",
            )
            # Bez cancels=: anulowanie zamykało generator, więc po Stop nie było
            # pakowania ZIP. Handler tylko ustawia flagę i wraca od razu - proces
            # zabija pętla logu treningu (patrz TrainingMonitor.request_stop).
            stop_btn.click(
                stop_training_ui,
                inputs=model_name,
                outputs=stop_status,
                queue=False,
            )

            resume.change(_on_resume_change, inputs=resume, outputs=resume_box).then(
                _refresh_drive_ckpts,
                inputs=[resume, model_name],
                outputs=[drive_ckpt, drive_ckpt_info],
            )
            drive_ckpt_refresh.click(
                _refresh_drive_ckpts,
                inputs=[resume, model_name],
                outputs=[drive_ckpt, drive_ckpt_info],
            )
            model_name.blur(
                _refresh_drive_ckpts,
                inputs=[resume, model_name],
                outputs=[drive_ckpt, drive_ckpt_info],
            )
            drive_retry_btn.click(
                _retry_drive_save,
                inputs=[train_zip_state, train_name_state, model_name],
                outputs=save_log_box,
            )

        with gr.Tab("3 · Zamień głos"):
            gr.Markdown(
                "Model nakłada wyuczoną barwę na inne nagranie. "
                "Melodia i tekst zostają, zmienia się głos.",
                elem_classes="rvc-intro",
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, elem_classes="rvc-card"):
                    _step(
                        1,
                        "Wgraj model",
                        "ZIP z zakładki 2 albo sam plik .pth.",
                    )
                    model = gr.File(
                        label="Model ZIP lub .pth", type="filepath", elem_id="rvc_infer_model"
                    )
                    with gr.Accordion(
                        "Osobny plik .index", open=False, elem_id="rvc_index_acc"
                    ):
                        index_file = gr.File(
                            label="Plik .index - tylko jeśli nie ma go w ZIP",
                            type="filepath",
                            file_types=[".index"],
                            elem_id="rvc_index_file",
                        )

                    _step(2, "Wgraj nagranie do zamiany", "wav, mp3, flac, ogg, m4a, aac.")
                    # File zamiast Audio: widać postęp uploadu, bez pustego 00:00.
                    source = gr.File(
                        label="Nagranie źródłowe",
                        type="filepath",
                        file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
                        elem_id="rvc_source",
                    )

                    _step(
                        3,
                        "Dopasuj wysokość",
                        "0 = bez zmian. Głos męski → żeński zwykle +12, odwrotnie −12.",
                    )
                    pitch = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        step=1,
                        value=0,
                        label="Zmiana wysokości głosu (półtony)",
                        elem_id="rvc_pitch",
                    )
                    with gr.Accordion(
                        "Ustawienia zaawansowane",
                        open=False,
                        elem_id="rvc_f0_acc",
                    ):
                        infer_f0_method = gr.Dropdown(
                            choices=list(F0_METHODS),
                            value=DEFAULT_F0_METHOD,
                            label="Metoda odczytu wysokości głosu",
                            elem_id="rvc_infer_f0",
                            info="rmvpe - zalecana. GPU/CPU wybierane automatycznie.",
                        )
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.05,
                            value=DEFAULT_INDEX_RATE,
                            label="Udział indeksu",
                            elem_id="rvc_index_rate",
                            info="0 = sam model, 1 = maksymalnie barwa z próbek. Działa przy pliku .index.",
                        )
                        infer_embedder = gr.Dropdown(
                            choices=[EMBEDDER_FROM_MODEL, *EMBEDDER_MODELS],
                            value=EMBEDDER_FROM_MODEL,
                            label="Model cech mowy",
                            elem_id="rvc_infer_embedder",
                            info=(
                                "„Z modelu” bierze wybór zapisany przy treningu. "
                                f"Inny niż {DEFAULT_EMBEDDER_MODEL} tylko wtedy, "
                                "gdy tak trenowano."
                            ),
                        )
                    infer_btn = gr.Button(
                        "Zamień głos", variant="primary", size="lg", elem_id="rvc_infer_btn"
                    )

                with gr.Column(scale=2, elem_classes="rvc-card"):
                    _step(4, "Odsłuchaj i pobierz", "Wynik pojawi się tu po konwersji.")
                    # Ukryty do czasu wyniku - bez pustego prostokąta 00:00 w trakcie.
                    out_audio = gr.Audio(
                        label="Wynik",
                        type="filepath",
                        show_download_button=True,
                        visible=False,
                        elem_id="rvc_out_audio",
                    )
                    out_download = gr.File(
                        label="Pobierz wynik WAV",
                        type="filepath",
                        elem_id="rvc_out_download",
                    )
                    infer_next = gr.Markdown(visible=False, elem_classes="rvc-next")
                    _block_title("Dziennik konwersji", "rvc_infer_log_title")
                    infer_log = gr.HTML(value=_log_html(""), elem_id="rvc_infer_log")
                    gr.Markdown(
                        "Używaj tylko nagrań, do których masz prawo - własnych "
                        "lub za zgodą osoby nagranej.",
                        elem_classes="rvc-note",
                    )

            infer_outputs = [out_audio, out_download, infer_log, infer_next]
            infer_btn.click(
                _infer_ui_stream,
                inputs=[
                    model,
                    index_file,
                    source,
                    pitch,
                    infer_f0_method,
                    index_rate,
                    infer_embedder,
                ],
                outputs=infer_outputs,
                scroll_to_output=False,
                show_progress="minimal",
            )
            source.change(
                _on_infer_input_ui,
                inputs=source,
                outputs=infer_outputs,
            )
            model.change(
                _on_infer_input_ui,
                inputs=model,
                outputs=infer_outputs,
            )

        gr.Markdown(
            "Silnik: [Applio](https://github.com/IAHispano/Applio) licencja MIT · "
            "Wymaga skonfigurowanego środowiska Applio - patrz `SETUP_APPLIO.md` · "
            "Projekt inżynierski - Klaudiusz Waśko",
            elem_classes="rvc-footer",
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("APP_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("APP_PORT", "7860")),
        share=os.environ.get("APP_SHARE", "0") == "1",
        i18n=_GRADIO_I18N,
    )


if __name__ == "__main__":
    main()
