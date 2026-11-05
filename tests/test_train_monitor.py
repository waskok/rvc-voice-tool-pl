"""Testy: odczyt strat z TensorBoard, monitor postępu/stop, pary G/D na Drive, podsumowanie próbek."""

from __future__ import annotations

import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc_pipeline.checkpoints import list_checkpoint_pairs, prune_checkpoint_pairs
from rvc_pipeline.dataset_info import estimate_training_seconds, summarize_samples
from rvc_pipeline.tb_scalars import ScalarReader
from rvc_pipeline.train_monitor import TrainingMonitor, _kill_tree, format_duration


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(payload)) + payload


def _event(step: int, values: dict[str, float]) -> bytes:
    summary = b"".join(
        _ld(1, _ld(1, tag.encode()) + _varint(2 << 3 | 5) + struct.pack("<f", v))
        for tag, v in values.items()
    )
    return _varint(1 << 3 | 1) + struct.pack("<d", 0.0) + _varint(2 << 3) + _varint(step) + _ld(5, summary)


def _record(data: bytes) -> bytes:
    return struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4


def test_scalar_reader_incremental_and_partial_record(tmp_path: Path):
    f = tmp_path / "events.out.tfevents.1.host"
    first = _record(_event(41, {"loss_avg_50/g/total": 38.5, "other": 1.0}))
    second = _record(_event(82, {"loss_avg_50/g/total": 37.0}))
    f.write_bytes(first + second[:10])
    reader = ScalarReader(tmp_path, {"loss_avg_50/g/total"})
    assert reader.read_new() == [("loss_avg_50/g/total", 41, 38.5)]
    with open(f, "ab") as fh:
        fh.write(second[10:])
    assert reader.read_new() == [("loss_avg_50/g/total", 82, 37.0)]
    assert reader.read_new() == []


def _epoch_line(epoch: int, step: int, speed: str) -> str:
    return f"glos | epoch={epoch} | step={step} | time=10:00:00 | training_speed={speed}"


def test_monitor_progress_eta_and_losses(tmp_path: Path):
    model_dir = tmp_path / "logs" / "glos"
    (model_dir / "eval").mkdir(parents=True)
    (model_dir / "eval" / "events.out.tfevents.1.x").write_bytes(
        _record(_event(41, {"loss_avg_50/g/total": 40.0, "loss_avg_50/g/mel": 20.0}))
    )
    mon = TrainingMonitor("glos")
    mon.configure(model_dir, total_epochs=100, save_every=25, backup_dir=None, resume_epoch=None)
    mon.training_started = True
    speeds = ["0:02:30", "0:02:28", "0:01:12", "0:01:11", "0:01:11", "0:01:10"]
    for i, sp in enumerate(speeds, start=1):
        assert mon.feed(_epoch_line(i, i * 41, sp))
    mon.feed(" 37%|███▋      | 15/41 [00:25<00:44,  1.71s/it]")
    assert mon.steps_per_epoch == 41
    assert mon.pace_seconds() == 71.0
    html = mon.progress_html()
    assert "Epoka <b>6</b> / 100" in html and "krok 15/41 w epoce 7" in html
    assert "1 h 50 min" in html  # (100 - 6 - 15/41) * 71 s ≈ 6648 s
    rows = mon.loss_rows()
    assert {r["seria"] for r in rows} == {"Generator: strata łączna", "Mel: różnica widma"}
    assert rows[0]["epoka"] == 1.0


def test_monitor_ignores_tqdm_before_training():
    mon = TrainingMonitor("glos")
    mon.feed(" 50%|█████     | 5/10 [00:01<00:01]")
    assert mon.in_epoch is None


def test_kill_tree_stops_children():
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    from rvc_pipeline.train_monitor import _children_map

    child = _children_map()[proc.pid][0]
    _kill_tree(proc.pid, grace_s=5)
    proc.wait(timeout=5)
    time.sleep(0.2)
    status = Path(f"/proc/{child}/stat")
    assert not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_pairs_listing_and_pruning(tmp_path: Path):
    for e in (25, 50, 75):
        (tmp_path / f"G_glos_e{e}.pth").write_bytes(b"g")
        (tmp_path / f"D_glos_e{e}.pth").write_bytes(b"d")
    (tmp_path / "G_glos_e100.pth").write_bytes(b"g")  # bez pary D
    pairs = list_checkpoint_pairs(tmp_path)
    assert [p.epoch for p in pairs] == [75, 50, 25]
    assert pairs[0].label.startswith("epoka 75")
    prune_checkpoint_pairs(tmp_path, keep=2)
    assert [p.epoch for p in list_checkpoint_pairs(tmp_path)] == [75, 50]
    assert (tmp_path / "G_glos_e100.pth").exists()


def _wav(path: Path, seconds: float, sr: int = 16000) -> Path:
    sf.write(str(path), np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


def test_summary_two_zips_and_one_wav(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for zname in ("a.zip", "b.zip"):
        with zipfile.ZipFile(tmp_path / zname, "w") as zf:
            for i in range(3):
                w = _wav(src / f"sample_{i:04d}.wav", 4.0)
                zf.write(w, w.name)
    single = _wav(tmp_path / "extra.wav", 6.0)
    text = summarize_samples(
        [str(tmp_path / "a.zip"), str(tmp_path / "b.zip"), str(single)], epochs=100, batch_size=12
    )
    assert "2 ZIP-y + 1 plik(i) audio" in text
    assert "**7 plików**" in text and "0 min 30 s" in text
    assert "Bardzo mało materiału" in text


def test_estimate_matches_own_measurement():
    pace, total = estimate_training_seconds(13.0, 100)
    assert pace == 71.0
    assert format_duration(total) == "2 h 19 min"
