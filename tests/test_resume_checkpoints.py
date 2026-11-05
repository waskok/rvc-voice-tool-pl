"""Testy wznawiania treningu, zapisu G/D na Drive i łączenia kilku ZIP-ów."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

from rvc_pipeline import checkpoints
from rvc_pipeline.applio_train import (
    ApplioTrainingError,
    _install_uploaded_checkpoints,
    _prepare_dataset_dir,
    latest_training_checkpoint,
    run_training,
    save_checkpoint_to_drive,
    save_model_to_drive,
)


@pytest.fixture
def fake_applio(tmp_path: Path, monkeypatch) -> Path:
    applio_dir = tmp_path / "Applio"
    applio_dir.mkdir()
    (applio_dir / "core.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("APPLIO_DIR", str(applio_dir))
    monkeypatch.setenv("APPLIO_PYTHON", sys.executable)
    return applio_dir


@pytest.fixture
def fake_drive(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "MyDrive"
    root.mkdir()
    monkeypatch.setattr(checkpoints, "DRIVE_ROOT", root)
    monkeypatch.setattr(checkpoints, "DRIVE_MODELS_DIR", root / "RVC_Colab_Models")
    return root


def _zip_with(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, f"audio-{path.stem}-{n}")
    return path


def test_two_zips_with_same_sample_names_do_not_overwrite(tmp_path: Path):
    a = _zip_with(tmp_path / "nagranie_a.zip", ["sample_0001.wav", "sample_0002.wav"])
    b = _zip_with(tmp_path / "nagranie_b.zip", ["sample_0001.wav"])
    dataset = _prepare_dataset_dir([str(a), str(b)], tmp_path / "work")
    names = sorted(p.name for p in dataset.iterdir())
    assert names == [
        "nagranie_b_sample_0001.wav",
        "sample_0001.wav",
        "sample_0002.wav",
    ]
    assert (dataset / "sample_0001.wav").read_text() == "audio-nagranie_a-sample_0001.wav"


def test_latest_checkpoint_matches_applio_digit_sort(tmp_path: Path):
    for n in ("G_500.pth", "G_4100.pth", "D_4100.pth", "voice_100e_4100s.pth"):
        (tmp_path / n).write_bytes(b"x")
    assert latest_training_checkpoint(tmp_path, "G").name == "G_4100.pth"
    assert latest_training_checkpoint(tmp_path, "d").name == "D_4100.pth"
    assert latest_training_checkpoint(tmp_path / "brak", "G") is None


def test_install_uploaded_checkpoints_replaces_old_pair(tmp_path: Path):
    model_dir = tmp_path / "logs" / "glos"
    model_dir.mkdir(parents=True)
    (model_dir / "G_2333333.pth").write_bytes(b"old-g")
    (model_dir / "D_123.pth").write_bytes(b"old-d")
    g = tmp_path / "G_glos_e100.pth"
    d = tmp_path / "D_glos_e100.pth"
    g.write_bytes(b"new-g")
    d.write_bytes(b"new-d")
    _install_uploaded_checkpoints(model_dir, str(g), str(d))
    assert sorted(p.name for p in model_dir.iterdir()) == ["D_2333333.pth", "G_2333333.pth"]
    assert latest_training_checkpoint(model_dir, "G").read_bytes() == b"new-g"


def test_new_training_refuses_to_silently_resume(fake_applio: Path):
    model_dir = fake_applio / "logs" / "glos"
    model_dir.mkdir(parents=True)
    (model_dir / "G_2333333.pth").write_bytes(b"g")
    with pytest.raises(ApplioTrainingError, match="Wznów trening"):
        next(run_training("glos", ["x.wav"]))


def test_resume_without_checkpoint_or_upload_fails(fake_applio: Path):
    with pytest.raises(ApplioTrainingError, match="Brak checkpointu"):
        next(run_training("glos", [], resume=True))


def test_resume_requires_both_g_and_d(fake_applio: Path, tmp_path: Path):
    g = tmp_path / "G.pth"
    g.write_bytes(b"g")
    with pytest.raises(ApplioTrainingError, match="oba checkpointy"):
        next(run_training("glos", ["x.wav"], resume=True, resume_g_path=str(g)))


def test_save_every_out_of_range(fake_applio: Path):
    with pytest.raises(ApplioTrainingError, match="dozwolone"):
        next(run_training("glos", ["x.wav"], save_every=150))


def test_save_checkpoint_to_drive_copies_latest(fake_applio: Path, fake_drive: Path):
    model_dir = fake_applio / "logs" / "glos"
    model_dir.mkdir(parents=True)
    (model_dir / "D_2333333.pth").write_bytes(b"d" * 10)
    msg = save_checkpoint_to_drive("glos", "D")
    saved = list((fake_drive / "RVC_Colab_Models" / "glos" / "checkpoints").iterdir())
    assert len(saved) == 1 and saved[0].name.startswith("D_glos")
    assert saved[0].read_bytes() == b"d" * 10
    assert "zapisano checkpoint dyskryminatora" in msg


def test_save_checkpoint_to_drive_without_checkpoint(fake_applio: Path, fake_drive: Path):
    assert "Brak checkpointu generatora" in save_checkpoint_to_drive("glos", "G")


def test_save_model_to_drive(tmp_path: Path, fake_drive: Path):
    zip_path = tmp_path / "glos.zip"
    zip_path.write_bytes(b"zip")
    msg = save_model_to_drive(str(zip_path), "Mój Głos")
    saved = list((fake_drive / "RVC_Colab_Models").glob("Moj_Glos_*.zip"))
    assert len(saved) == 1 and "zapisano model" in msg
    assert "brak gotowego modelu" in save_model_to_drive(None, "glos")
