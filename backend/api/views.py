import hashlib
import json
import logging

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView

from coaches.feedback_generator import generate_rep_feedback
from coaches.super_form_codes_config import SUPER_FORM_CODES_CONFIG

from calibration_store import store

from .models import WorkoutSessionRecord
from .serializers import (
    AskQuestionSerializer,
    FormAnalysisRequestSerializer,
    SaveSessionSerializer,
    SessionDataSerializer,
    WorkoutSessionRecordSerializer,
)
from .services import (
    get_cached_llm_summary,
    get_root_payload,
    llm_coach,
    log_llm_event,
    persist_form_analysis,
    save_session_payload,
    save_workout_payload,
    set_cached_llm_summary,
)


logger = logging.getLogger(__name__)


class RootApiView(APIView):
    def get(self, _request):
        return Response(get_root_payload())


class SummaryApiView(APIView):
    def post(self, request):
        serializer = SessionDataSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        cache_key = "llm:summary:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        summary = get_cached_llm_summary(cache_key)
        if summary is None:
            summary = llm_coach.generate_session_summary(payload)
            set_cached_llm_summary(cache_key, summary)
        log_llm_event("session_summary", payload, summary)
        return Response({"summary": summary})


class AskApiView(APIView):
    def post(self, request):
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        answer = llm_coach.answer_question(payload["session_data"], payload["question"])
        log_llm_event(
            "question_answer",
            payload,
            answer,
            question=payload["question"],
        )
        return Response({"answer": answer})


class SaveWorkoutApiView(APIView):
    def post(self, request):
        serializer = SessionDataSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        try:
            result = save_workout_payload(str(payload.get("exercise", "workout")), payload)
            return Response(
                {
                    "status": "success",
                    "message": f"Workout saved to {result['path']}",
                    "session_id": str(result["record"].id),
                }
            )
        except Exception as exc:
            logger.error("Failed to save workout: %s", exc)
            return Response(
                {"status": "error", "message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SaveSessionApiView(APIView):
    def post(self, request):
        serializer = SaveSessionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        try:
            result = save_session_payload(payload)
            return Response(
                {
                    "status": "success",
                    "message": f"Session saved to {result['session']}",
                    "session_id": str(result["record"].id),
                }
            )
        except Exception as exc:
            logger.error("Failed to save session: %s", exc)
            return Response(
                {"status": "error", "message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnalyzeFormApiView(APIView):
    def post(self, request):
        serializer = FormAnalysisRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data

        try:
            exercise = payload["exercise"]
            form_snapshots = payload["form_snapshots"]
            total_reps = payload.get("total_reps") or len(form_snapshots)
            exercise_super_codes = SUPER_FORM_CODES_CONFIG.get(exercise, {})

            super_form_codes_count = {}
            good_reps = 0
            processed_snapshots = []

            for snapshot in form_snapshots:
                super_codes = snapshot.get("form_states", [])
                static_codes = snapshot.get("static_primitives", {})
                dynamic_codes = snapshot.get("dynamic_primitives", {})

                for super_code in super_codes:
                    super_form_codes_count[super_code] = super_form_codes_count.get(super_code, 0) + 1
                if "GOOD_REP" in super_codes:
                    good_reps += 1

                feedback = generate_rep_feedback(
                    exercise=exercise,
                    static_form_codes=static_codes,
                    dynamic_form_codes=dynamic_codes,
                    super_form_codes=super_codes,
                )

                enriched_snapshot = dict(snapshot)
                enriched_snapshot["feedback"] = feedback
                processed_snapshots.append(enriched_snapshot)

            issues = {key: value for key, value in super_form_codes_count.items() if key != "GOOD_REP"}
            sorted_issues = sorted(issues.items(), key=lambda item: item[1], reverse=True)

            top_issues = []
            for super_code_name, count in sorted_issues[:3]:
                description = exercise_super_codes.get(super_code_name, {}).get("description", "")
                top_issues.append(
                    {
                        "super_form_code": super_code_name,
                        "count": count,
                        "percentage": round((count / total_reps) * 100, 1) if total_reps > 0 else 0,
                        "description": description,
                    }
                )

            score = round((good_reps / total_reps) * 100, 1) if total_reps > 0 else 0
            analysis = {
                "exercise": exercise,
                "total_reps": total_reps,
                "good_reps": good_reps,
                "score": score,
                "super_form_codes_count": super_form_codes_count,
                "top_issues": top_issues,
                "snapshots": processed_snapshots,
            }

            session = None
            if payload.get("session_id"):
                session = WorkoutSessionRecord.objects.filter(id=payload["session_id"]).first()
            analysis_record = persist_form_analysis(payload, analysis, session=session)

            return Response(
                {
                    "status": "success",
                    "analysis": analysis,
                    "analysis_id": str(analysis_record.id),
                }
            )
        except Exception as exc:
            logger.error("Form analysis failed: %s", exc)
            return Response(
                {"status": "error", "message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class WorkoutSessionListApiView(generics.ListAPIView):
    serializer_class = WorkoutSessionRecordSerializer

    def get_queryset(self):
        queryset = WorkoutSessionRecord.objects.prefetch_related("sets").all()
        session_type = self.request.query_params.get("session_type")
        exercise = self.request.query_params.get("exercise")
        if session_type:
            queryset = queryset.filter(session_type=session_type)
        if exercise:
            queryset = queryset.filter(exercise=exercise)
        return queryset


class WorkoutSessionDetailApiView(generics.RetrieveAPIView):
    serializer_class = WorkoutSessionRecordSerializer
    queryset = WorkoutSessionRecord.objects.prefetch_related("sets").all()


class CalibrationSummaryApiView(APIView):
    def get(self, _request, exercise):
        summary = store.to_summary(exercise)
        return Response({"exercise": exercise, **summary})
