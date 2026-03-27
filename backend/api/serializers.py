from rest_framework import serializers

from .models import (
    CalibrationPreference,
    CalibrationRecordModel,
    FormAnalysisRecord,
    LLMInteraction,
    WorkoutSessionRecord,
    WorkoutSetRecord,
)


class SessionDataSerializer(serializers.Serializer):
    total_reps = serializers.IntegerField()
    success_rate = serializers.FloatField()
    mistakes = serializers.JSONField()
    avg_tempo = serializers.FloatField()
    exercise = serializers.CharField()
    session_details = serializers.JSONField(required=False)
    form_analysis = serializers.JSONField(required=False)
    rep_durations = serializers.JSONField(required=False)


class AskQuestionSerializer(serializers.Serializer):
    session_data = serializers.JSONField()
    question = serializers.CharField()


class FormAnalysisRequestSerializer(serializers.Serializer):
    exercise = serializers.CharField()
    form_snapshots = serializers.ListField(child=serializers.JSONField())
    total_reps = serializers.IntegerField(required=False, default=0)
    session_id = serializers.UUIDField(required=False)


class WorkoutSetPayloadSerializer(serializers.Serializer):
    exercise = serializers.CharField(required=False, allow_blank=True)
    target_reps = serializers.IntegerField(required=False)
    completed_reps = serializers.IntegerField(required=False)
    remaining_reps = serializers.IntegerField(required=False)
    is_complete = serializers.BooleanField(required=False)
    mistakes = serializers.JSONField(required=False)
    form_snapshots = serializers.ListField(child=serializers.JSONField(), required=False)
    duration_seconds = serializers.FloatField(required=False, allow_null=True)


class SaveSessionSerializer(serializers.Serializer):
    id = serializers.CharField(required=False)
    name = serializers.CharField(required=False, allow_blank=True)
    sets = serializers.ListField(child=serializers.JSONField(), required=False)
    current_set_index = serializers.IntegerField(required=False)
    is_complete = serializers.BooleanField(required=False)
    total_sets = serializers.IntegerField(required=False)
    completed_sets = serializers.IntegerField(required=False)
    total_reps_completed = serializers.IntegerField(required=False)
    total_reps_target = serializers.IntegerField(required=False)
    all_mistakes = serializers.JSONField(required=False)
    started_at = serializers.FloatField(required=False, allow_null=True)
    finished_at = serializers.FloatField(required=False, allow_null=True)
    duration_seconds = serializers.FloatField(required=False, allow_null=True)


class CalibrationRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = CalibrationRecordModel
        fields = [
            "id",
            "exercise",
            "mode",
            "timestamp",
            "angles",
            "eta",
            "canonical",
            "critic",
            "images",
            "created_at",
        ]


class CalibrationPreferenceSerializer(serializers.ModelSerializer):
    active = serializers.SerializerMethodField()
    critics = serializers.JSONField()
    records = serializers.SerializerMethodField()

    class Meta:
        model = CalibrationPreference
        fields = ["exercise", "active", "critics", "records"]

    def get_active(self, obj):
        return {
            "common": str(obj.active_common_record_id) if obj.active_common_record_id else None,
            "calibration": str(obj.active_calibration_record_id) if obj.active_calibration_record_id else None,
        }

    def get_records(self, obj):
        records = CalibrationRecordModel.objects.filter(exercise=obj.exercise).order_by("-timestamp", "-created_at")
        return CalibrationRecordSerializer(records, many=True).data


class WorkoutSetRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkoutSetRecord
        fields = [
            "order_index",
            "exercise",
            "target_reps",
            "completed_reps",
            "remaining_reps",
            "is_complete",
            "mistakes",
            "form_snapshots",
            "duration_seconds",
        ]


class WorkoutSessionRecordSerializer(serializers.ModelSerializer):
    sets = WorkoutSetRecordSerializer(many=True, read_only=True)

    class Meta:
        model = WorkoutSessionRecord
        fields = [
            "id",
            "session_type",
            "name",
            "exercise",
            "total_reps",
            "success_rate",
            "avg_tempo",
            "total_reps_target",
            "total_reps_completed",
            "total_sets",
            "completed_sets",
            "duration_seconds",
            "mistakes",
            "session_details",
            "form_analysis",
            "created_at",
            "sets",
        ]


class FormAnalysisRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = FormAnalysisRecord
        fields = [
            "id",
            "session",
            "exercise",
            "total_reps",
            "good_reps",
            "score",
            "super_form_codes_count",
            "top_issues",
            "snapshots",
            "created_at",
        ]


class LLMInteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMInteraction
        fields = [
            "id",
            "session",
            "interaction_type",
            "question",
            "input_payload",
            "response_text",
            "log_path",
            "created_at",
        ]
