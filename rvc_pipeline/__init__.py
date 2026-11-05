"""Pipeline RVC: przygotowanie datasetu, trening, inferencja.

Integracja z wybraną bazą open-source (MIT) + własna warstwa konfiguracji/UI.
"""

from rvc_pipeline.pipeline import list_presets, run_infer_ui, run_train_ui

__all__ = ["run_infer_ui", "list_presets", "run_train_ui"]
