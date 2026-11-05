"""Historia zapisów (Dysk Google, pliki do pobrania) widoczna w Zakładce 2.

Wpisy tylko przybywają - automatyczne kopie z wątków w tle i ręczne
ponowienia trafiają do jednej listy, żeby było widać, co i kiedy się
zapisało. Jedna lista na proces UI (Colab = jeden użytkownik).
"""

from __future__ import annotations

import threading
from datetime import datetime

try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo("Europe/Warsaw")
except Exception:  # brak bazy stref czasowych - czas lokalny maszyny
    _TZ = None

_MAX_ENTRIES = 500
_entries: list[str] = []
_lock = threading.Lock()
_version = 0


def now_hm(fmt: str = "%H:%M:%S") -> str:
    return datetime.now(_TZ).strftime(fmt)


def record(message: str) -> str:
    """Dopisuje wpis z godziną i zwraca go."""
    global _version
    entry = f"{now_hm()}  {message}"
    with _lock:
        _entries.append(entry)
        del _entries[:-_MAX_ENTRIES]
        _version += 1
    return entry


def text() -> str:
    with _lock:
        if not _entries:
            return "Brak zapisów w tej sesji."
        return "\n".join(_entries)


def version() -> int:
    return _version


def clear() -> None:
    global _version
    with _lock:
        _entries.clear()
        _version += 1
