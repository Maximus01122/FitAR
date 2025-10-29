import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "calibrations.json"
DEFAULT_CRITIC = 0.2


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


@dataclass
class CalibrationRecord:
    id: str
    exercise: str
    mode: str
    timestamp: str
    angles: Dict[str, float]
    eta: Dict[str, float]
    canonical: Dict[str, float]
    critic: float
    images: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "exercise": self.exercise,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "angles": self.angles,
            "eta": self.eta,
            "canonical": self.canonical,
            "critic": self.critic,
            "images": self.images,
        }


class CalibrationStore:
    def __init__(self, path: Path = DEFAULT_STORE_PATH):
        self.path = path
        self.data: Dict[str, Any] = {
            "exercises": {},
        }
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {"exercises": {}}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def _ensure_exercise(self, exercise: str):
        exercises = self.data.setdefault("exercises", {})
        if exercise not in exercises:
            exercises[exercise] = {
                "records": [],
                "active": {"common": None, "calibration": None},
                "critics": {"common": DEFAULT_CRITIC, "calibration": DEFAULT_CRITIC},
            }

    def list_records(self, exercise: str) -> List[Dict[str, Any]]:
        self._ensure_exercise(exercise)
        return list(self.data["exercises"][exercise]["records"])

    def add_record(self, record: CalibrationRecord):
        exercise = record.exercise
        self._ensure_exercise(exercise)
        records = self.data["exercises"][exercise]["records"]
        records = [r for r in records if r["id"] != record.id]
        records.append(record.to_dict())
        # Sort newest first
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        self.data["exercises"][exercise]["records"] = records
        self.set_active_record(exercise, record.mode, record.id, save=False)
        self._save()

    def set_active_record(
        self, exercise: str, mode: str, record_id: Optional[str], save: bool = True
    ):
        self._ensure_exercise(exercise)
        if record_id is not None:
            records = self.data["exercises"][exercise]["records"]
            if not any(rec["id"] == record_id for rec in records):
                return False
        self.data["exercises"][exercise]["active"][mode] = record_id
        if save:
            self._save()
        return True

    def get_active_record(self, exercise: str, mode: str) -> Optional[Dict[str, Any]]:
        self._ensure_exercise(exercise)
        record_id = self.data["exercises"][exercise]["active"].get(mode)
        if not record_id:
            return None
        for record in self.data["exercises"][exercise]["records"]:
            if record["id"] == record_id:
                return record
        return None

    def set_critic(self, exercise: str, mode: str, critic: float):
        self._ensure_exercise(exercise)
        self.data["exercises"][exercise]["critics"][mode] = critic
        self._save()

    def delete_record(self, exercise: str, record_id: str) -> bool:
        self._ensure_exercise(exercise)
        info = self.data["exercises"][exercise]
        original_len = len(info["records"])
        info["records"] = [r for r in info["records"] if r["id"] != record_id]
        if original_len == len(info["records"]):
            return False
        for mode in ("common", "calibration"):
            if info["active"].get(mode) == record_id:
                info["active"][mode] = None
        self._save()
        return True

    def get_critics(self, exercise: str) -> Dict[str, float]:
        self._ensure_exercise(exercise)
        return dict(self.data["exercises"][exercise]["critics"])

    def to_summary(self, exercise: str) -> Dict[str, Any]:
        self._ensure_exercise(exercise)
        info = self.data["exercises"][exercise]
        return {
            "records": list(info["records"]),
            "active": dict(info["active"]),
            "critics": dict(info["critics"]),
        }


store = CalibrationStore()


def new_record(
    exercise: str,
    mode: str,
    angles: Dict[str, float],
    eta: Dict[str, float],
    canonical: Dict[str, float],
    critic: float,
    images: Optional[Dict[str, Optional[str]]] = None,
) -> CalibrationRecord:
    return CalibrationRecord(
        id=str(uuid.uuid4()),
        exercise=exercise,
        mode=mode,
        timestamp=_timestamp(),
        angles=angles,
        eta=eta,
        canonical=canonical,
        critic=critic,
        images=images or {},
    )
