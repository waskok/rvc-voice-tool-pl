"""Checkpointy treningowe G/D: wyszukiwanie w ``logs/``, kopie na Google Drive, lista par.

Układ na Drive (wymaga zamontowania w komórce notatnika Colab):
    MyDrive/RVC_Colab_Models/<model>.zip                         - gotowe modele
    MyDrive/RVC_Colab_Models/<model>/checkpoints/G_<model>_e<N>.pth
    MyDrive/RVC_Colab_Models/<model>/checkpoints/D_<model>_e<N>.pth

Para G + D z tej samej epoki N wystarcza do wznowienia treningu w nowej
sesji (Applio wczytuje oba pliki w ``rvc/train/train.py``).
"""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_MODELS_DIR = DRIVE_ROOT / "RVC_Colab_Models"

_EPOCH_IN_NAME = re.compile(r"_e(\d+)\.pth$", re.IGNORECASE)


def latest_training_checkpoint(model_dir: Path, kind: str) -> Path | None:
    """Najnowszy ``G_*.pth`` / ``D_*.pth`` - ta sama reguła co Applio
    ``latest_checkpoint_path`` (sortowanie po cyfrach w nazwie, bez podkatalogów)."""
    prefix = kind.upper()
    if not model_dir.is_dir():
        return None
    candidates = [p for p in model_dir.glob(f"{prefix}_*.pth") if p.is_file()]
    if not candidates:
        return None
    return max(
        candidates, key=lambda p: int("".join(filter(str.isdigit, p.name)) or 0)
    )


def wait_for_checkpoint_written(
    model_dir: Path,
    since_ts: float,
    timeout_s: float = 180.0,
    quiet_s: float = 5.0,
    poll_s: float = 1.0,
) -> bool:
    """Czeka, aż Applio dopisze G i D (zmienione po ``since_ts``) i ucichnie zapis.

    Applio najpierw wypisuje linię epoki, potem zapisuje G, D i mały model
    ``.pth``. Kopiowanie albo zabicie procesu w środku zapisu dałoby uszkodzony
    plik - przy ``--save-only-latest`` jedyny checkpoint. „Cisza” = żaden
    ``.pth`` w katalogu nie zmienił się przez ``quiet_s`` sekund.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        g = latest_training_checkpoint(model_dir, "G")
        d = latest_training_checkpoint(model_dir, "D")
        if g and d:
            try:
                fresh = (
                    g.stat().st_mtime >= since_ts - 0.5
                    and d.stat().st_mtime >= since_ts - 0.5
                )
                newest = max(p.stat().st_mtime for p in model_dir.glob("*.pth"))
            except (OSError, ValueError):
                fresh, newest = False, time.time()
            if fresh and time.time() - newest >= quiet_s:
                return True
        time.sleep(poll_s)
    return False


def drive_mounted() -> bool:
    return DRIVE_ROOT.is_dir()


def drive_dest_dir(subdir: str | None = None) -> tuple[Path | None, str]:
    """Katalog docelowy na już zamontowanym Drive albo ``(None, komunikat)``.

    ``google.colab.drive.mount`` musi być wywołane w **komórce notebooka**
    (interaktywne jądro IPython). Z callbacku Gradio mount pada z
    ``'NoneType' object has no attribute 'kernel'`` - dlatego tu **nie**
    wołamy ``drive.mount``, tylko kopiujemy do już podpiętego
    ``/content/drive/MyDrive``.
    """
    if not DRIVE_ROOT.is_dir():
        try:
            import google.colab  # type: ignore  # noqa: F401
            in_colab = True
        except ImportError:
            in_colab = False

        if in_colab:
            return None, (
                "Google Drive: dysk nie jest podłączony. Zatrzymaj komórkę ze "
                "startem UI, uruchom „Podłącz Google Drive”, potem znów start UI. "
                "Pliki w /content zostają. Trwający trening zostanie przerwany - "
                "wznowisz go z checkpointu w tej samej sesji."
            )
        return None, (
            "Google Drive: dostępny tylko w Google Colab po podłączeniu dysku "
            "w komórce notatnika - pominięto zapis."
        )

    dest_dir = DRIVE_MODELS_DIR / subdir if subdir else DRIVE_MODELS_DIR
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, f"Google Drive: nie udało się utworzyć {dest_dir}: {exc}"
    return dest_dir, ""


def drive_checkpoint_dir(model_name: str) -> Path:
    return DRIVE_MODELS_DIR / model_name / "checkpoints"


def checkpoint_drive_name(kind: str, model_name: str, epoch: int | None) -> str:
    if epoch is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{kind.upper()}_{model_name}_{stamp}.pth"
    return f"{kind.upper()}_{model_name}_e{epoch}.pth"


def copy_atomic(src: Path, dest: Path) -> None:
    """Kopia przez plik tymczasowy - przerwana kopia nie zostawi „prawie” checkpointu."""
    tmp = dest.with_name(f".{dest.name}.part")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


@dataclass(frozen=True)
class CheckpointPair:
    g_path: Path
    d_path: Path
    epoch: int | None

    @property
    def label(self) -> str:
        epoch = f"epoka {self.epoch}" if self.epoch is not None else "epoka ?"
        return f"{epoch} - {self.g_path.name} + {self.d_path.name}"


def list_checkpoint_pairs(ckpt_dir: Path) -> list[CheckpointPair]:
    """Pary G/D o tej samej nazwie (poza prefiksem), od najnowszej epoki."""
    if not ckpt_dir.is_dir():
        return []
    pairs = []
    for g in ckpt_dir.glob("G_*.pth"):
        d = g.with_name("D_" + g.name[2:])
        if not d.is_file():
            continue
        m = _EPOCH_IN_NAME.search(g.name)
        pairs.append(CheckpointPair(g, d, int(m.group(1)) if m else None))
    pairs.sort(
        key=lambda p: (p.epoch if p.epoch is not None else -1, p.g_path.stat().st_mtime),
        reverse=True,
    )
    return pairs


def prune_checkpoint_pairs(ckpt_dir: Path, keep: int) -> list[Path]:
    """Usuwa starsze pary z numerem epoki, zostawia ``keep`` najnowszych."""
    removed = []
    numbered = [p for p in list_checkpoint_pairs(ckpt_dir) if p.epoch is not None]
    for pair in numbered[keep:]:
        for path in (pair.g_path, pair.d_path):
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                pass
    return removed
