from django.contrib import admin

from .models import (
    CalibrationPreference,
    CalibrationRecordModel,
    FormAnalysisRecord,
    LLMInteraction,
    WorkoutSessionRecord,
    WorkoutSetRecord,
)


@admin.register(CalibrationRecordModel)
class CalibrationRecordAdmin(admin.ModelAdmin):
    list_display = ("exercise", "mode", "critic", "timestamp", "created_at")
    list_filter = ("exercise", "mode")
    search_fields = ("exercise", "id", "legacy_source")
    ordering = ("-created_at",)


@admin.register(CalibrationPreference)
class CalibrationPreferenceAdmin(admin.ModelAdmin):
    list_display = ("exercise", "active_common_record", "active_calibration_record", "updated_at")
    search_fields = ("exercise",)


class WorkoutSetInline(admin.TabularInline):
    model = WorkoutSetRecord
    extra = 0
    readonly_fields = (
        "order_index",
        "exercise",
        "target_reps",
        "completed_reps",
        "remaining_reps",
        "is_complete",
        "duration_seconds",
    )


@admin.register(WorkoutSessionRecord)
class WorkoutSessionRecordAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "session_type",
        "exercise",
        "total_reps",
        "success_rate",
        "created_at",
    )
    list_filter = ("session_type", "exercise")
    search_fields = ("name", "exercise", "legacy_source")
    inlines = [WorkoutSetInline]


@admin.register(FormAnalysisRecord)
class FormAnalysisRecordAdmin(admin.ModelAdmin):
    list_display = ("exercise", "score", "total_reps", "good_reps", "created_at")
    list_filter = ("exercise",)
    search_fields = ("exercise",)


@admin.register(LLMInteraction)
class LLMInteractionAdmin(admin.ModelAdmin):
    list_display = ("interaction_type", "question", "created_at")
    list_filter = ("interaction_type",)
    search_fields = ("question", "response_text")
