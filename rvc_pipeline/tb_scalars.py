"""Przyrostowy odczyt wartości skalarnych z plików TensorBoard (``events.out.tfevents.*``).

Applio zapisuje straty treningu wyłącznie do TensorBoard
(``logs/<model>/eval/``, ``SummaryWriter.add_scalar`` w ``rvc/train/train.py``)
- w logu tekstowym jest tylko najniższa wartość od początku. Nasz proces UI
nie ma ``tensorboard``/``torch``, więc format czytamy ręcznie:

* plik = ciąg rekordów TFRecord: ``uint64 długość | uint32 crc | dane | uint32 crc``,
* dane = komunikat protobuf ``Event`` (pole 2 ``step``, pole 5 ``summary``),
* ``Summary.value`` (pole 1) = ``tag`` (1) + ``simple_value`` (2, float32)
  albo ``tensor`` (8) z ``float_val`` / ``double_val``.

Rekord niedokończony (plik w trakcie zapisu) jest pomijany i doczytywany
przy następnym wywołaniu.

Źródła:
  - TFRecord: https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details
  - ``event.proto`` / ``summary.proto`` / ``tensor.proto``:
    https://github.com/tensorflow/tensorboard/tree/master/tensorboard/compat/proto
  - Kodowanie protobuf (varint, wire types): https://protobuf.dev/programming-guides/encoding/
"""

from __future__ import annotations

import struct
from pathlib import Path


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _fields(buf: bytes):
    """Iteruje ``(numer_pola, wire_type, wartość)`` po komunikacie protobuf."""
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            value = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Nieobsługiwany wire type {wire}")
        yield field, wire, value


def _tensor_value(buf: bytes) -> float | None:
    for field, wire, value in _fields(buf):
        if field == 5:  # float_val
            if wire == 2 and len(value) >= 4:
                return struct.unpack("<f", value[:4])[0]
            if wire == 5:
                return struct.unpack("<f", value)[0]
        if field == 6:  # double_val
            if wire == 2 and len(value) >= 8:
                return struct.unpack("<d", value[:8])[0]
            if wire == 1:
                return struct.unpack("<d", value)[0]
    return None


def _parse_event(data: bytes) -> list[tuple[str, int, float]]:
    step = 0
    summary = None
    for field, _wire, value in _fields(data):
        if field == 2:
            step = value
        elif field == 5:
            summary = value
    if summary is None:
        return []
    out = []
    for field, _wire, value in _fields(summary):
        if field != 1:
            continue
        tag = None
        scalar = None
        for vf, vwire, vval in _fields(value):
            if vf == 1:
                tag = vval.decode("utf-8", "replace")
            elif vf == 2 and vwire == 5:
                scalar = struct.unpack("<f", vval)[0]
            elif vf == 8 and vwire == 2:
                scalar = _tensor_value(vval)
        if tag is not None and scalar is not None:
            out.append((tag, int(step), float(scalar)))
    return out


class ScalarReader:
    """Czyta nowe skalary z katalogu z plikami tfevents (zapamiętuje pozycję w plikach)."""

    def __init__(self, log_dir: Path, tags: set[str] | None = None):
        self.log_dir = Path(log_dir)
        self.tags = tags
        self._offsets: dict[Path, int] = {}

    def read_new(self) -> list[tuple[str, int, float]]:
        """Zwraca nowe ``(tag, step, wartość)`` od poprzedniego wywołania."""
        if not self.log_dir.is_dir():
            return []
        out: list[tuple[str, int, float]] = []
        for path in sorted(self.log_dir.glob("events.out.tfevents.*")):
            offset = self._offsets.get(path, 0)
            try:
                with open(path, "rb") as fh:
                    fh.seek(offset)
                    blob = fh.read()
            except OSError:
                continue
            pos = 0
            while pos + 12 <= len(blob):
                (length,) = struct.unpack("<Q", blob[pos:pos + 8])
                end = pos + 12 + length + 4
                if end > len(blob):
                    break
                data = blob[pos + 12:pos + 12 + length]
                pos = end
                try:
                    events = _parse_event(data)
                except (ValueError, IndexError, struct.error):
                    continue
                for tag, step, value in events:
                    if self.tags is None or tag in self.tags:
                        out.append((tag, step, value))
            self._offsets[path] = offset + pos
        return out
