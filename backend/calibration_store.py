import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CRITIC = 0.2


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_models() -> Tuple[Any, Any]:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fitcoach_backend.settings")
    import django
    from django.apps import apps

    if not apps.ready:
        django.setup()
    return apps.get_model("api", "CalibrationPreference"), apps.get_model("api", "CalibrationRecordModel")


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
    def _ensure_exercise(self, exercise: str):
        CalibrationPreference, _ = _get_models()
        preference, _ = CalibrationPreference.objects.get_or_create(
            exercise=exercise,
            defaults={"critics": {"common": DEFAULT_CRITIC, "calibration": DEFAULT_CRITIC}},
        )
        critics = dict(preference.critics or {})
        critics.setdefault("common", DEFAULT_CRITIC)
        critics.setdefault("calibration", DEFAULT_CRITIC)
        if critics != preference.critics:
            preference.critics = critics
            preference.save(update_fields=["critics", "updated_at"])
        return preference

    def list_records(self, exercise: str) -> List[Dict[str, Any]]:
        _, CalibrationRecordModel = _get_models()
        self._ensure_exercise(exercise)
        return [self._model_to_dict(record) for record in CalibrationRecordModel.objects.filter(exercise=exercise)]

    def add_record(self, record: CalibrationRecord):
        CalibrationPreference, CalibrationRecordModel = _get_models()
        preference = self._ensure_exercise(record.exercise)
        record_model, _ = CalibrationRecordModel.objects.update_or_create(
            id=record.id,
            defaults={
                "exercise": record.exercise,
                "mode": record.mode,
                "timestamp": record.timestamp,
                "angles": record.angles,
                "eta": record.eta,
                "canonical": record.canonical,
                "critic": record.critic,
                "images": record.images,
            },
        )
        field_name = self._active_field_name(record.mode)
        setattr(preference, field_name, record_model)
        preference.save(update_fields=[field_name, "updated_at"])

    def set_active_record(self, exercise: str, mode: str, record_id: Optional[str], save: bool = True):
        CalibrationPreference, CalibrationRecordModel = _get_models()
        preference = self._ensure_exercise(exercise)
        field_name = self._active_field_name(mode)
        if record_id is not None:
            try:
                record = CalibrationRecordModel.objects.get(id=record_id, exercise=exercise)
            except CalibrationRecordModel.DoesNotExist:
                return False
            setattr(preference, field_name, record)
        else:
            setattr(preference, field_name, None)
        if save:
            preference.save(update_fields=[field_name, "updated_at"])
        return True

    def get_active_record(self, exercise: str, mode: str) -> Optional[Dict[str, Any]]:
        preference = self._ensure_exercise(exercise)
        field_name = self._active_field_name(mode)
        record = getattr(preference, field_name)
        if not record:
            return None
        return self._model_to_dict(record)

    def set_critic(self, exercise: str, mode: str, critic: float):
        preference = self._ensure_exercise(exercise)
        critics = dict(preference.critics or {})
        critics[mode] = critic
        preference.critics = critics
        preference.save(update_fields=["critics", "updated_at"])

    def delete_record(self, exercise: str, record_id: str) -> bool:
        CalibrationPreference, CalibrationRecordModel = _get_models()
        preference = self._ensure_exercise(exercise)
        try:
            record = CalibrationRecordModel.objects.get(id=record_id, exercise=exercise)
        except CalibrationRecordModel.DoesNotExist:
            return False
        if preference.active_common_record_id == record.id:
            preference.active_common_record = None
        if preference.active_calibration_record_id == record.id:
            preference.active_calibration_record = None
        preference.save(update_fields=["active_common_record", "active_calibration_record", "updated_at"])
        record.delete()
        return True

    def get_critics(self, exercise: str) -> Dict[str, float]:
        preference = self._ensure_exercise(exercise)
        critics = dict(preference.critics or {})
        critics.setdefault("common", DEFAULT_CRITIC)
        critics.setdefault("calibration", DEFAULT_CRITIC)
        return critics

    def to_summary(self, exercise: str) -> Dict[str, Any]:
        preference = self._ensure_exercise(exercise)
        return {
            "records": self.list_records(exercise),
            "active": {
                "common": str(preference.active_common_record_id) if preference.active_common_record_id else None,
                "calibration": str(preference.active_calibration_record_id) if preference.active_calibration_record_id else None,
            },
            "critics": self.get_critics(exercise),
        }

    def _active_field_name(self, mode: str) -> str:
        return "active_calibration_record" if mode == "calibration" else "active_common_record"

    def _model_to_dict(self, record_model) -> Dict[str, Any]:
        return {
            "id": str(record_model.id),
            "exercise": record_model.exercise,
            "mode": record_model.mode,
            "timestamp": record_model.timestamp,
            "angles": record_model.angles,
            "eta": record_model.eta,
            "canonical": record_model.canonical,
            "critic": record_model.critic,
            "images": record_model.images,
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
