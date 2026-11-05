"""Poprawki UX treningu: strumień logu, Stop, patch Colab, wykres strat."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from rvc_pipeline.applio_train import (
    _APPLIO_TRAIN_PATCH_MARK,
    _patch_applio_train_for_colab,
    _stream_step,
    _unbuffered_cmd,
)
from rvc_pipeline.train_monitor import TrainingMonitor, TrainingStopped, loss_chart_svg

PY = sys.executable


def _script(code: str) -> list[str]:
    return [PY, "-c", code]


def test_unbuffered_cmd_inserts_dash_u():
    assert _unbuffered_cmd(["/venv/bin/python", "core.py", "train"]) == [
        "/venv/bin/python",
        "-u",
        "core.py",
        "train",
    ]


_PRISTINE_DATALOADER = (
    "from torch.utils.data import DataLoader\n"
    "save_every_weights = _strtobool(sys.argv[10])\n"
    "train_loader = DataLoader(\n"
    "    train_dataset,\n"
    "    num_workers=4,\n"
    "    shuffle=False,\n"
    "    persistent_workers=True,\n"
    "    prefetch_factor=8,\n"
    ")\n"
)


def _train_py(tmp_path: Path, text: str) -> Path:
    train_py = tmp_path / "rvc" / "train" / "train.py"
    train_py.parent.mkdir(parents=True)
    train_py.write_text(text, encoding="utf-8")
    return train_py


def test_patch_keeps_applio_dataloader_and_disables_mid_run_extract(tmp_path: Path):
    train_py = _train_py(tmp_path, _PRISTINE_DATALOADER)
    _patch_applio_train_for_colab(tmp_path)
    text = train_py.read_text(encoding="utf-8")
    assert "save_every_weights = False" in text
    assert "    num_workers=4,\n" in text
    assert "    persistent_workers=True,\n" in text
    assert "    prefetch_factor=8,\n" in text
    _patch_applio_train_for_colab(tmp_path)
    assert train_py.read_text(encoding="utf-8") == text


def test_patch_restores_dataloader_changed_by_v1_and_v2(tmp_path: Path):
    train_py = _train_py(
        tmp_path,
        "from torch.utils.data import DataLoader  # RVC_COLAB_PL_PATCH_V2\n"
        f"_strtobool(sys.argv[10])  # CLI ignorowane - {_APPLIO_TRAIN_PATCH_MARK}\n"
        "save_every_weights = False  # bez mid-run extract; final model zostaje\n"
        "train_loader = DataLoader(\n"
        "    num_workers=0,  # RVC_COLAB_PL_PATCH_V2\n"
        "    persistent_workers=False,  # RVC_COLAB_PL_PATCH_V2\n"
        "    # prefetch_factor=8,  # RVC_COLAB_PL_PATCH_V2: tylko przy workers>0\n"
        ")\n"
        f"other = DataLoader(num_workers=2,  # {_APPLIO_TRAIN_PATCH_MARK}\n)\n",
    )
    _patch_applio_train_for_colab(tmp_path)
    text = train_py.read_text(encoding="utf-8")
    assert "RVC_COLAB_PL_PATCH_V2" not in text
    assert "from torch.utils.data import DataLoader\n" in text
    assert "    num_workers=4,\n" in text
    assert "other = DataLoader(num_workers=4,\n" in text
    assert "    persistent_workers=True,\n" in text
    assert "    prefetch_factor=8,\n" in text
    assert text.count("save_every_weights = False") == 1


def test_stream_shows_epoch_line_before_silence():
    # Pasek tqdm (\r) + linia epoki, potem cisza (jak przy zapisie G/D).
    # Poprzednia wersja (select + readline) pokazywała „41/41”, a linia epoki
    # utknęła w buforze TextIOWrapper do kolejnych danych z procesu.
    code = (
        "import sys, time\n"
        "for ep in range(1, 4):\n"
        "    for i in range(1, 42):\n"
        "        sys.stdout.write(f'\\r{i * 100 // 41}%|#| {i}/41 [00:01<00:00]')\n"
        "        sys.stdout.flush()\n"
        "        time.sleep(0.002)\n"
        "    sys.stdout.write('\\n')\n"
        "    sys.stdout.write(f'm | epoch={ep} | step={ep * 41} | time=0:00:01 | "
        "training_speed=0:00:01\\n')\n"
        "    sys.stdout.flush()\n"
        "time.sleep(5)\n"
        "print('end')\n"
    )
    start = time.monotonic()
    seen_at = None
    for log in _stream_step(_script(code), cwd=Path.cwd(), timeout_s=30, step_name="t"):
        if seen_at is None and "epoch=3" in log:
            seen_at = time.monotonic() - start
    assert seen_at is not None and seen_at < 2.5


def test_stream_throttles_yields_and_collapses_tqdm():
    code = (
        "import sys, time\n"
        "for i in range(1, 2001):\n"
        "    sys.stdout.write(f'\\r{i * 100 // 2000}%|##| {i}/2000 [00:01<00:00]')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.001)\n"
        "print()\n"
        "print('done')\n"
    )
    logs = list(_stream_step(_script(code), cwd=Path.cwd(), timeout_s=60, step_name="t"))
    assert len(logs) < 40
    final = logs[-1]
    assert "done" in final
    assert "2000/2000" in final
    assert final.splitlines()[-2:] == ["100%|##| 2000/2000 [00:01<00:00]", "done"]
    assert "1999/2000" not in final


def test_stop_kills_process_and_raises_training_stopped():
    monitor = TrainingMonitor("x")
    code = "import time\nprint('start', flush=True)\ntime.sleep(120)\n"
    gen = _stream_step(_script(code), cwd=Path.cwd(), timeout_s=300, step_name="t", monitor=monitor)
    start = time.monotonic()
    with pytest.raises(TrainingStopped):
        for log in gen:
            if "start" in log and not monitor.stop_requested:
                msg = monitor.request_stop()
                assert "Zatrzymuję" in msg
    assert time.monotonic() - start < 20


def test_stop_waits_for_checkpoint_save_before_kill():
    monitor = TrainingMonitor("x")
    monitor._saving_epoch = 5
    code = "import time\nprint('start', flush=True)\ntime.sleep(120)\n"
    gen = _stream_step(_script(code), cwd=Path.cwd(), timeout_s=300, step_name="t", monitor=monitor)
    released_at: list[float] = []

    def release():
        time.sleep(2.5)
        released_at.append(time.monotonic())
        monitor._saving_epoch = None

    threading.Thread(target=release, daemon=True).start()
    with pytest.raises(TrainingStopped):
        for log in gen:
            if "start" in log and not monitor.stop_requested:
                assert "zapis checkpointu" in monitor.request_stop()
    assert released_at and time.monotonic() >= released_at[0]


def test_next_epoch_progress_ends_checkpoint_save_pending(tmp_path: Path):
    m = TrainingMonitor("x")
    m.configure(tmp_path, total_epochs=20, save_every=5, backup_dir=None, resume_epoch=None)
    m.training_started = True
    m.feed("x | epoch=5 | step=55 | time=0:00:14 | training_speed=0:00:14")
    assert m.checkpoint_save_pending
    m.feed(" 9%|#  | 1/11 [00:01<00:10,  1.00s/it]")
    assert not m.checkpoint_save_pending


def test_closing_stream_kills_process():
    monitor = TrainingMonitor("x")
    code = "import time\nprint('start', flush=True)\ntime.sleep(120)\n"
    gen = _stream_step(_script(code), cwd=Path.cwd(), timeout_s=300, step_name="t", monitor=monitor)
    for log in gen:
        if "start" in log:
            break
    proc = monitor._proc
    assert proc is not None
    gen.close()
    assert proc.poll() is not None


def test_progress_html_keeps_percent_separate_from_epoch():
    m = TrainingMonitor("x")
    m.configure(
        Path("/tmp"), total_epochs=15, save_every=5, backup_dir=None, resume_epoch=None
    )
    m.training_started = True
    m.feed(
        "x | epoch=4 | step=44 | time=0:00:14 | training_speed=0:00:14"
    )
    m.feed("11/11 [00:14<00:00,  1.26s/it]")
    html = m.progress_html()
    assert "rvc-progress-pct" in html
    assert "533%" not in html


def test_progress_html_warns_on_long_silence():
    m = TrainingMonitor("x")
    m.configure(
        Path("/tmp"), total_epochs=55, save_every=50, backup_dir=None, resume_epoch=None
    )
    m.training_started = True
    m.feed("x | epoch=2 | step=82 | time=0:01:20 | training_speed=0:01:20")
    m.last_log_at = time.time() - 120
    html = m.progress_html()
    assert "Brak nowych linii" in html


def test_loss_chart_svg_placeholder_and_lines():
    assert "<svg" not in loss_chart_svg([])
    rows = [
        {"epoka": e / 2, "strata": 30 - e, "seria": s}
        for e in range(10)
        for s in ("Generator: strata łączna", "Mel: różnica widma")
    ]
    svg = loss_chart_svg(rows)
    assert svg.count('class="rvc-chart-panel"') == 2
    assert svg.count("<polyline") == 4
    assert "Mel: różnica widma" in svg
