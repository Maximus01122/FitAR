import uuid

from django.db import models


def default_critics():
    return {"common": 0.2, "calibration": 0.2}


class CalibrationRecordModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    exercise = models.CharField(max_length=64, db_index=True)
    mode = models.CharField(max_length=32, default="common")
    timestamp = models.CharField(max_length=64, db_index=True)
    angles = models.JSONField(default=dict)
    eta = models.JSONField(default=dict)
    canonical = models.JSONField(default=dict)
    critic = models.FloatField(default=0.2)
    images = models.JSONField(default=dict)
    legacy_source = models.CharField(max_length=255, blank=True, default="", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp", "-created_at"]
        indexes = [
            models.Index(fields=["exercise", "mode"]),
        ]

    def __str__(self):
        return f"{self.exercise}:{self.mode}:{self.id}"


class CalibrationPreference(models.Model):
    exercise = models.CharField(max_length=64, unique=True)
    critics = models.JSONField(default=default_critics)
    active_common_record = models.ForeignKey(
        CalibrationRecordModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="active_common_for",
    )
    active_calibration_record = models.ForeignKey(
        CalibrationRecordModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="active_calibration_for",
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.exercise


class WorkoutSessionRecord(models.Model):
    SESSION_TYPES = (
        ("workout", "Workout"),
        ("session", "Session"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_type = models.CharField(max_length=16, choices=SESSION_TYPES)
    name = models.CharField(max_length=255, blank=True, default="")
    exercise = models.CharField(max_length=128, blank=True, default="", db_index=True)
    total_reps = models.IntegerField(default=0)
    success_rate = models.FloatField(default=0.0)
    avg_tempo = models.FloatField(default=0.0)
    total_reps_target = models.IntegerField(default=0)
    total_reps_completed = models.IntegerField(default=0)
    total_sets = models.IntegerField(default=0)
    completed_sets = models.IntegerField(default=0)
    duration_seconds = models.FloatField(default=0.0)
    mistakes = models.JSONField(default=dict)
    session_details = models.JSONField(default=dict)
    form_analysis = models.JSONField(default=dict)
    raw_payload = models.JSONField(default=dict)
    legacy_source = models.CharField(max_length=255, blank=True, default="", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session_type", "created_at"]),
            models.Index(fields=["exercise", "created_at"]),
        ]

    def __str__(self):
        label = self.name or self.exercise or self.session_type
        return f"{label} ({self.session_type})"


class WorkoutSetRecord(models.Model):
    session = models.ForeignKey(
        WorkoutSessionRecord,
        on_delete=models.CASCADE,
        related_name="sets",
    )
    order_index = models.PositiveIntegerField()
    exercise = models.CharField(max_length=64)
    target_reps = models.IntegerField(default=0)
    completed_reps = models.IntegerField(default=0)
    remaining_reps = models.IntegerField(default=0)
    is_complete = models.BooleanField(default=False)
    mistakes = models.JSONField(default=dict)
    form_snapshots = models.JSONField(default=list)
    duration_seconds = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["order_index"]
        unique_together = [("session", "order_index")]

    def __str__(self):
        return f"{self.session_id}:{self.order_index}:{self.exercise}"


class FormAnalysisRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        WorkoutSessionRecord,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="form_analyses",
    )
    exercise = models.CharField(max_length=64, db_index=True)
    total_reps = models.IntegerField(default=0)
    good_reps = models.IntegerField(default=0)
    score = models.FloatField(default=0.0)
    super_form_codes_count = models.JSONField(default=dict)
    top_issues = models.JSONField(default=list)
    snapshots = models.JSONField(default=list)
    raw_request = models.JSONField(default=dict)
    raw_analysis = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.exercise}:{self.score}"


class LLMInteraction(models.Model):
    INTERACTION_TYPES = (
        ("session_summary", "Session Summary"),
        ("question_answer", "Question Answer"),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        WorkoutSessionRecord,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="llm_interactions",
    )
    interaction_type = models.CharField(max_length=32, choices=INTERACTION_TYPES)
    question = models.TextField(blank=True, default="")
    input_payload = models.JSONField(default=dict)
    response_text = models.TextField()
    log_path = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.interaction_type}:{self.id}"
