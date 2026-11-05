"""Udział indeksu i model cech przy konwersji."""

from __future__ import annotations

import pytest

from rvc_pipeline.applio_engine import (
    ApplioInferenceError,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_INDEX_RATE,
    EMBEDDER_FROM_MODEL,
    format_index_rate,
    normalize_index_rate,
    resolve_embedder,
)


def test_index_rate_default_and_bounds():
    assert normalize_index_rate(None) == DEFAULT_INDEX_RATE
    assert normalize_index_rate("") == DEFAULT_INDEX_RATE
    assert normalize_index_rate(0) == 0
    assert normalize_index_rate(1) == 1
    assert normalize_index_rate("0.75") == 0.75
    assert format_index_rate(0.75) == "0.75"
    assert format_index_rate(1.0) == "1"
    with pytest.raises(ApplioInferenceError, match="0 do 1"):
        normalize_index_rate(1.5)
    with pytest.raises(ApplioInferenceError, match="0 do 1"):
        normalize_index_rate("dużo")


def test_embedder_from_model_uses_value_stored_at_training():
    name, note = resolve_embedder(EMBEDDER_FROM_MODEL, "spin")
    assert name == "spin"
    assert "spin" in note
    assert "treningu" in note


def test_explicit_choice_overrides_file():
    name, note = resolve_embedder("chinese-hubert-base", "contentvec")
    assert name == "chinese-hubert-base"
    assert "zakładce 3" in note


def test_missing_or_unknown_stored_embedder_falls_back():
    name, note = resolve_embedder("z modelu", None)
    assert name == DEFAULT_EMBEDDER_MODEL
    assert "nie ma zapisanego" in note
    name, note = resolve_embedder(None, "custom-path")
    assert name == DEFAULT_EMBEDDER_MODEL
    assert "nieznany" in note


def test_unknown_choice_is_rejected():
    with pytest.raises(ApplioInferenceError, match="Nieznany model cech"):
        resolve_embedder("hubert", "contentvec")
