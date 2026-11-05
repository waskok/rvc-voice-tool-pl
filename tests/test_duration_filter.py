"""Testy filtra długości Zakładki 1 (bez I/O / bez librosa.load)."""

from __future__ import annotations

import numpy as np

from audio_tools.process import (
    DatasetSummary,
    _apply_duration_filter,
    _split_fixed_windows,
)


def test_split_fixed_windows_covers_full_signal():
    sr = 40000
    audio = np.ones(int(25.0 * sr), dtype=np.float32)
    parts = _split_fixed_windows(audio, sr, 12.0)
    assert len(parts) == 3  # 12 + 12 + 1
    assert sum(len(p) for p in parts) == len(audio)


def test_long_chunk_is_resplit_not_rejected():
    """Wyłączenie cięcia po ciszy + 161 s nie może wyzerować datasetu."""
    sr = 40000
    audio = np.ones(int(161.0 * sr), dtype=np.float32)  # ~2:41
    kept, rejected_short, resplit_long = _apply_duration_filter(
        [audio], sr, min_duration_s=1.0, max_duration_s=12.0
    )
    assert rejected_short == 0  # 161 / 12 = 13×12 + 5 → 5 s ≥ 1.0
    assert len(kept) == 14
    assert resplit_long == 14
    total_s = sum(len(c) / sr for c in kept)
    assert abs(total_s - 161.0) < 1e-3


def test_short_chunks_still_rejected():
    sr = 40000
    chunks = [
        np.ones(int(0.4 * sr), dtype=np.float32),
        np.ones(int(2.0 * sr), dtype=np.float32),
    ]
    kept, rejected_short, resplit_long = _apply_duration_filter(
        chunks, sr, min_duration_s=1.0, max_duration_s=12.0
    )
    assert rejected_short == 1
    assert len(kept) == 1
    assert resplit_long == 0


def test_summary_mentions_resplit_not_reject_long():
    text = DatasetSummary(accepted=14, rejected_short=0, resplit_long=14).as_text()
    assert "Podzielone z za długich" in text
    assert "Odrzucone (za długie" not in text
