import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from django.core.cache import cache

from coaches import FormAnalyzer, LLMCoach
from pose_backends import build_pose_backend, get_available_backends

from .models import FormAnalysisRecord, LLMInteraction, WorkoutSessionRecord, WorkoutSetRecord


logger = logging.getLogger(__name__)

POSE_BACKEND_NAME = os.getenv("POSE_BACKEND", "mediapipe_2d")
BASE_DIR = Path(__file__).resolve().parent.parent
WORKOUTS_DIR = BASE_DIR / "workouts"
LLM_LOGS_DIR = BASE_DIR / "llm_logs"

WORKOUTS_DIR.mkdir(exist_ok=True)
LLM_LOGS_DIR.mkdir(exist_ok=True)

CEREBRAS_API_KEY = os.environ.get(
    "CEREBRAS_API_KEY",
    "csk-pe9ve2dc58528hxp34jwd4v8jk426t4mk9223my3j3k6ej5c",
)

llm_coach = LLMCoach(api_key=CEREBRAS_API_KEY)
form_analyzer = FormAnalyzer()


def get_root_payload() -> Dict[str, Any]:
    return {
        "message": "Welcome to FitCoachAR - Real-Time Adaptive Exercise Coaching API",
        "pose_backend": POSE_BACKEND_NAME,
        "available_backends": get_available_backends(),
    }


def build_active_backend():
    return build_pose_backend(POSE_BACKEND_NAME)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=4))


def log_llm_event(
    kind: str,
    input_payload: Dict[str, Any],
    output_payload: Any,
    *,
    session: Optional[WorkoutSessionRecord] = None,
    question: str = "",
) -> Path:
    timestamp = timestamp_slug()
    filename = "summary" if kind == "session_summary" else "qa"
    path = LLM_LOGS_DIR / f"{filename}_{timestamp}.json"
    write_json(
        path,
        {
            "type": kind,
            "timestamp": timestamp,
            "input_to_llm": input_payload,
            "llm_response": output_payload,
        },
    )
    LLMInteraction.objects.create(
        session=session,
        interaction_type=kind,
        question=question,
        input_payload=input_payload,
        response_text=str(output_payload),
        log_path=str(path),
    )
    logger.info("LLM log saved to %s", path)
    return path


def _resolve_exercise(payload: Dict[str, Any]) -> str:
    exercise = str(payload.get("exercise", "")).strip()
    if exercise:
        return exercise
    sets = payload.get("sets") or []
    if sets and isinstance(sets, list):
        return str(sets[0].get("exercise", ""))
    return ""


def persist_session_record(payload: Dict[str, Any], session_type: str) -> WorkoutSessionRecord:
    exercise = _resolve_exercise(payload)
    name = str(payload.get("name") or exercise or session_type).strip()
    total_reps = int(payload.get("total_reps", payload.get("total_reps_completed", 0)) or 0)
    success_rate = float(payload.get("success_rate", 0.0) or 0.0)
    avg_tempo = float(payload.get("avg_tempo", 0.0) or 0.0)
    total_reps_target = int(payload.get("total_reps_target", 0) or 0)
    total_reps_completed = int(payload.get("total_reps_completed", total_reps) or 0)
    total_sets = int(payload.get("total_sets", len(payload.get("sets", []) or [])) or 0)
    completed_sets = int(payload.get("completed_sets", 0) or 0)
    duration_seconds = float(payload.get("duration_seconds", 0.0) or 0.0)
    mistakes = payload.get("mistakes") or payload.get("all_mistakes") or {}
    session_details = payload.get("session_details") or payload
    form_analysis = payload.get("form_analysis") or {}

    record = WorkoutSessionRecord.objects.create(
        session_type=session_type,
        name=name,
        exercise=exercise,
        total_reps=total_reps,
        success_rate=success_rate,
        avg_tempo=avg_tempo,
        total_reps_target=total_reps_target,
        total_reps_completed=total_reps_completed,
        total_sets=total_sets,
        completed_sets=completed_sets,
        duration_seconds=duration_seconds,
        mistakes=mistakes,
        session_details=session_details,
        form_analysis=form_analysis,
        raw_payload=payload,
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

    return record


def save_workout_payload(exercise: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = WORKOUTS_DIR / f"{exercise}_{timestamp_slug()}.json"
    write_json(path, payload)
    record = persist_session_record(payload, session_type="workout")
    return {"path": path, "record": record}


def save_session_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = timestamp_slug()
    session_name = str(payload.get("name", "session")).replace(" ", "_")
    session_path = WORKOUTS_DIR / f"session_{session_name}_{timestamp}.json"
    log_path = LLM_LOGS_DIR / f"session_{timestamp}.json"
    write_json(session_path, payload)
    write_json(log_path, payload)
    record = persist_session_record(payload, session_type="session")
    logger.info("Session saved to %s", session_path)
    return {"session": session_path, "log": log_path, "record": record}


def persist_form_analysis(
    request_payload: Dict[str, Any],
    analysis: Dict[str, Any],
    *,
    session: Optional[WorkoutSessionRecord] = None,
) -> FormAnalysisRecord:
    record = FormAnalysisRecord.objects.create(
        session=session,
        exercise=str(analysis.get("exercise", request_payload.get("exercise", ""))),
        total_reps=int(analysis.get("total_reps", 0) or 0),
        good_reps=int(analysis.get("good_reps", 0) or 0),
        score=float(analysis.get("score", 0.0) or 0.0),
        super_form_codes_count=analysis.get("super_form_codes_count") or {},
        top_issues=analysis.get("top_issues") or [],
        snapshots=analysis.get("snapshots") or [],
        raw_request=request_payload,
        raw_analysis=analysis,
    )
    return record


def get_cached_llm_summary(cache_key: str):
    return cache.get(cache_key)


def set_cached_llm_summary(cache_key: str, value: str):
    cache.set(cache_key, value, timeout=300)
