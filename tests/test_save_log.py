"""Historia zapisów w kroku 5: automatyczny zapis modelu, pobieranie G/D."""

from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

from rvc_pipeline import checkpoints, save_log
from rvc_pipeline.applio_train import (
    _auto_save_model,
    prepare_checkpoint_downloads,
    retry_model_save,
    save_model_to_drive,
)


@pytest.fixture(autouse=True)
def _clean_log():
    save_log.clear()
    yield
    save_log.clear()


@pytest.fixture
def fake_applio(tmp_path: Path, monkeypatch) -> Path:
    applio_dir = tmp_path / "Applio"
    applio_dir.mkdir()
    (applio_dir / "core.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("APPLIO_DIR", str(applio_dir))
    monkeypatch.setenv("APPLIO_PYTHON", sys.executable)
    return applio_dir


@pytest.fixture
def no_drive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(checkpoints, "DRIVE_ROOT", tmp_path / "brak_dysku")
    monkeypatch.setattr(checkpoints, "DRIVE_MODELS_DIR", tmp_path / "brak_dysku" / "M")


@pytest.fixture
def fake_drive(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "MyDrive"
    root.mkdir()
    monkeypatch.setattr(checkpoints, "DRIVE_ROOT", root)
    monkeypatch.setattr(checkpoints, "DRIVE_MODELS_DIR", root / "RVC_Colab_Models")
    return root


def _model_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "glos.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("glos.pth", "pth")
    return zip_path


def test_log_keeps_every_entry_with_time():
    assert save_log.text() == "Brak zapisów w tej sesji."
    v0 = save_log.version()
    save_log.record("pierwszy")
    save_log.record("drugi")
    lines = save_log.text().splitlines()
    assert len(lines) == 2
    assert re.match(r"^\d{2}:\d{2}:\d{2}  pierwszy$", lines[0])
    assert lines[1].endswith("drugi")
    assert save_log.version() == v0 + 2


def test_auto_save_without_drive_is_logged_as_local_only(tmp_path: Path, no_drive):
    msg = _auto_save_model(_model_zip(tmp_path), "glos")
    assert "tylko lokalnie" in msg
    assert "Zapisz model na Dysku ponownie" in msg
    assert msg in save_log.text()


def test_auto_save_with_drive_copies_zip_and_logs_path(tmp_path: Path, fake_drive: Path):
    msg = _auto_save_model(_model_zip(tmp_path), "glos")
    saved = list((fake_drive / "RVC_Colab_Models").glob("glos_*.zip"))
    assert len(saved) == 1
    assert str(saved[0]) in msg
    assert msg in save_log.text()


def test_failed_retry_is_logged_too(no_drive):
    save_model_to_drive(None, "glos")
    assert "brak gotowego modelu" in save_log.text()


def test_retry_after_ui_restart_rebuilds_zip_from_logs(fake_applio: Path, fake_drive: Path):
    model_dir = fake_applio / "logs" / "glos"
    model_dir.mkdir(parents=True)
    (model_dir / "glos_100e_900s.pth").write_bytes(b"model")
    (model_dir / "glos.index").write_bytes(b"index")
    (model_dir / "G_2333333.pth").write_bytes(b"G")

    msg = retry_model_save(None, "glos")
    saved = list((fake_drive / "RVC_Colab_Models").glob("glos_*.zip"))
    assert len(saved) == 1 and str(saved[0]) in msg
    with zipfile.ZipFile(saved[0]) as zf:
        assert sorted(zf.namelist()) == ["glos.index", "glos_100e_900s.pth"]


def test_retry_without_any_model(fake_applio: Path, fake_drive: Path):
    assert "brak gotowego modelu „glos”" in retry_model_save(None, "glos")
    assert "brak gotowego modelu" in save_log.text()


def test_checkpoint_downloads_get_readable_names(fake_applio: Path):
    model_dir = fake_applio / "logs" / "glos"
    model_dir.mkdir(parents=True)
    (model_dir / "G_2333333.pth").write_bytes(b"G")
    (model_dir / "D_2333333.pth").write_bytes(b"D")

    g, d = prepare_checkpoint_downloads("glos", 50)
    assert g and d
    assert Path(g).name == checkpoints.checkpoint_drive_name("G", "glos", 50)
    assert Path(d).name == checkpoints.checkpoint_drive_name("D", "glos", 50)
    assert Path(g).read_bytes() == b"G"
    assert os.path.samefile(g, model_dir / "G_2333333.pth")

    g2, _ = prepare_checkpoint_downloads("glos", 100)
    assert not Path(g).exists()
    assert Path(g2).exists()


def test_checkpoint_downloads_missing(fake_applio: Path):
    assert prepare_checkpoint_downloads("glos", None) == (None, None)
