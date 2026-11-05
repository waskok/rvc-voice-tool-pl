"""Testy wyboru pliku .pth (inferencja vs checkpoint G_/D_)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rvc_pipeline.applio_engine import ApplioInferenceError, _pick_inference_pth
from rvc_pipeline.applio_train import (
    _ensure_applio_assets_config,
    _list_generator_ckpts,
    _list_infer_pths,
)


def test_pick_prefers_exported_over_g(tmp_path: Path):
    g = tmp_path / "G_225.pth"
    exp = tmp_path / "kw_model_15e_225s.pth"
    g.write_bytes(b"g")
    exp.write_bytes(b"export")
    chosen, _ = _pick_inference_pth([g, exp])
    assert chosen == exp


def test_pick_rejects_only_g(tmp_path: Path):
    g = tmp_path / "G_225.pth"
    g.write_bytes(b"g")
    with pytest.raises(ApplioInferenceError, match="checkpoint treningowy"):
        _pick_inference_pth([g])


def test_list_helpers_split_g_and_infer(tmp_path: Path):
    (tmp_path / "G_1.pth").write_bytes(b"1")
    (tmp_path / "D_1.pth").write_bytes(b"1")
    (tmp_path / "voice_15e_100s.pth").write_bytes(b"ok")
    assert [p.name for p in _list_infer_pths(tmp_path)] == ["voice_15e_100s.pth"]
    assert [p.name for p in _list_generator_ckpts(tmp_path)] == ["G_1.pth"]


def test_ensure_assets_config_creates_missing_file(tmp_path: Path):
    """Regresja: bez tego pliku Applio extract_model() cicho nie zapisuje
    małego modelu na końcu treningu (por. docstring w applio_train.py)."""
    cfg = tmp_path / "assets" / "config.json"
    assert not cfg.exists()
    _ensure_applio_assets_config(tmp_path)
    assert cfg.is_file()
    assert "model_author" in cfg.read_text(encoding="utf-8")


def test_ensure_assets_config_does_not_overwrite_existing(tmp_path: Path):
    cfg = tmp_path / "assets" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"model_author": "ktos"}', encoding="utf-8")
    _ensure_applio_assets_config(tmp_path)
    assert "ktos" in cfg.read_text(encoding="utf-8")
