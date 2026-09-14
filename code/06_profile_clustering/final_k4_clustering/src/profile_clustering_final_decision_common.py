from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

class ProfileClusteringFinalDecisionError(RuntimeError):
    pass

def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileClusteringFinalDecisionError(message)

def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path

def local_time_from_slot(slot: int) -> str:
    minutes = (int(slot) - 1) * 30
    return f'{minutes // 60:02d}:{minutes % 60:02d}'

def peak_band_2h(slot: int) -> str:
    start_slot = ((int(slot) - 1) // 4) * 4 + 1
    end_slot = min(start_slot + 3, 48)
    return f'{local_time_from_slot(start_slot)}–{local_time_from_slot(end_slot)}'
