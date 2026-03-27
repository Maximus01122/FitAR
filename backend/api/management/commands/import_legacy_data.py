import json
from pathlib import Path

from django.core.management.base import BaseCommand

from api.models import CalibrationPreference, CalibrationRecordModel, WorkoutSessionRecord, WorkoutSetRecord


class Command(BaseCommand):
    help = "Import legacy calibration/workout JSON files into Django models"

    def handle(self, *args, **options):
        base_dir = Path(__file__).resolve().parents[3]
        self.import_calibrations(base_dir / "calibrations.json")
        self.import_workouts(base_dir / "workouts")

    def import_calibrations(self, path: Path):
        if not path.exists():
            self.stdout.write("No legacy calibrations.json found; skipping calibrations import.")
            return

        payload = json.loads(path.read_text())
        exercises = payload.get("exercises", {})
        for exercise, info in exercises.items():
            preference, _ = CalibrationPreference.objects.get_or_create(exercise=exercise)
            critics = info.get("critics") or preference.critics
            created_ids = {}
            for record in info.get("records", []):
                model, _ = CalibrationRecordModel.objects.update_or_create(
                    id=record["id"],
                    defaults={
                        "exercise": exercise,
                        "mode": record.get("mode", "common"),
                        "timestamp": record.get("timestamp", ""),
                        "angles": record.get("angles", {}),
                        "eta": record.get("eta", {}),
                        "canonical": record.get("canonical", {}),
                        "critic": record.get("critic", 0.2),
                        "images": record.get("images", {}),
                        "legacy_source": str(path),
                    },
                )
                created_ids[str(model.id)] = model

            active = info.get("active", {})
            preference.critics = critics
            preference.active_common_record = created_ids.get(active.get("common"))
            preference.active_calibration_record = created_ids.get(active.get("calibration"))
            preference.save()
        self.stdout.write(self.style.SUCCESS("Imported legacy calibrations."))

    def import_workouts(self, directory: Path):
        if not directory.exists():
            self.stdout.write("No legacy workouts directory found; skipping workout import.")
            return

        for path in sorted(directory.glob("*.json")):
            if WorkoutSessionRecord.objects.filter(legacy_source=str(path)).exists():
                continue
            payload = json.loads(path.read_text())
            session_type = "session" if "sets" in payload else "workout"
            total_reps = int(payload.get("total_reps", payload.get("total_reps_completed", 0)) or 0)
            record = WorkoutSessionRecord.objects.create(
                session_type=session_type,
                name=str(payload.get("name") or payload.get("exercise") or path.stem),
                exercise=str(payload.get("exercise") or (payload.get("sets") or [{}])[0].get("exercise", "")),
                total_reps=total_reps,
                success_rate=float(payload.get("success_rate", 0.0) or 0.0),
                avg_tempo=float(payload.get("avg_tempo", 0.0) or 0.0),
                total_reps_target=int(payload.get("total_reps_target", 0) or 0),
                total_reps_completed=int(payload.get("total_reps_completed", total_reps) or 0),
                total_sets=int(payload.get("total_sets", len(payload.get("sets", []) or [])) or 0),
                completed_sets=int(payload.get("completed_sets", 0) or 0),
                duration_seconds=float(payload.get("duration_seconds", 0.0) or 0.0),
                mistakes=payload.get("mistakes") or payload.get("all_mistakes") or {},
                session_details=payload.get("session_details") or payload,
                form_analysis=payload.get("form_analysis") or {},
                raw_payload=payload,
                legacy_source=str(path),
            )
            for index, set_payload in enumerate(payload.get("sets", []) or []):
                WorkoutSetRecord.objects.create(
                    session=record,
                    order_index=index,
                    exercise=str(set_payload.get("exercise", "")),
                    target_reps=int(set_payload.get("target_reps", set_payload.get("reps", 0)) or 0),
                    completed_reps=int(set_payload.get("completed_reps", 0) or 0),
                    remaining_reps=int(set_payload.get("remaining_reps", 0) or 0),
                    is_complete=bool(set_payload.get("is_complete", False)),
                    mistakes=set_payload.get("mistakes") or {},
                    form_snapshots=set_payload.get("form_snapshots") or [],
                    duration_seconds=set_payload.get("duration_seconds"),
                )
        self.stdout.write(self.style.SUCCESS("Imported legacy workouts."))
