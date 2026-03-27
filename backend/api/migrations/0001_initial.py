from django.db import migrations, models
import django.db.models.deletion
import uuid


def default_critics():
    return {"common": 0.2, "calibration": 0.2}


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="CalibrationRecordModel",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("exercise", models.CharField(db_index=True, max_length=64)),
                ("mode", models.CharField(default="common", max_length=32)),
                ("timestamp", models.CharField(db_index=True, max_length=64)),
                ("angles", models.JSONField(default=dict)),
                ("eta", models.JSONField(default=dict)),
                ("canonical", models.JSONField(default=dict)),
                ("critic", models.FloatField(default=0.2)),
                ("images", models.JSONField(default=dict)),
                ("legacy_source", models.CharField(blank=True, db_index=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-timestamp", "-created_at"],
            },
        ),
        migrations.CreateModel(
            name="WorkoutSessionRecord",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("session_type", models.CharField(choices=[("workout", "Workout"), ("session", "Session")], max_length=16)),
                ("name", models.CharField(blank=True, default="", max_length=255)),
                ("exercise", models.CharField(blank=True, db_index=True, default="", max_length=128)),
                ("total_reps", models.IntegerField(default=0)),
                ("success_rate", models.FloatField(default=0.0)),
                ("avg_tempo", models.FloatField(default=0.0)),
                ("total_reps_target", models.IntegerField(default=0)),
                ("total_reps_completed", models.IntegerField(default=0)),
                ("total_sets", models.IntegerField(default=0)),
                ("completed_sets", models.IntegerField(default=0)),
                ("duration_seconds", models.FloatField(default=0.0)),
                ("mistakes", models.JSONField(default=dict)),
                ("session_details", models.JSONField(default=dict)),
                ("form_analysis", models.JSONField(default=dict)),
                ("raw_payload", models.JSONField(default=dict)),
                ("legacy_source", models.CharField(blank=True, db_index=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="LLMInteraction",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("interaction_type", models.CharField(choices=[("session_summary", "Session Summary"), ("question_answer", "Question Answer")], max_length=32)),
                ("question", models.TextField(blank=True, default="")),
                ("input_payload", models.JSONField(default=dict)),
                ("response_text", models.TextField()),
                ("log_path", models.CharField(blank=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("session", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="llm_interactions", to="api.workoutsessionrecord")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="FormAnalysisRecord",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("exercise", models.CharField(db_index=True, max_length=64)),
                ("total_reps", models.IntegerField(default=0)),
                ("good_reps", models.IntegerField(default=0)),
                ("score", models.FloatField(default=0.0)),
                ("super_form_codes_count", models.JSONField(default=dict)),
                ("top_issues", models.JSONField(default=list)),
                ("snapshots", models.JSONField(default=list)),
                ("raw_request", models.JSONField(default=dict)),
                ("raw_analysis", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("session", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="form_analyses", to="api.workoutsessionrecord")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="WorkoutSetRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("order_index", models.PositiveIntegerField()),
                ("exercise", models.CharField(max_length=64)),
                ("target_reps", models.IntegerField(default=0)),
                ("completed_reps", models.IntegerField(default=0)),
                ("remaining_reps", models.IntegerField(default=0)),
                ("is_complete", models.BooleanField(default=False)),
                ("mistakes", models.JSONField(default=dict)),
                ("form_snapshots", models.JSONField(default=list)),
                ("duration_seconds", models.FloatField(blank=True, null=True)),
                ("session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="sets", to="api.workoutsessionrecord")),
            ],
            options={
                "ordering": ["order_index"],
                "unique_together": {("session", "order_index")},
            },
        ),
        migrations.CreateModel(
            name="CalibrationPreference",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("exercise", models.CharField(max_length=64, unique=True)),
                ("critics", models.JSONField(default=default_critics)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("active_calibration_record", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="active_calibration_for", to="api.calibrationrecordmodel")),
                ("active_common_record", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="active_common_for", to="api.calibrationrecordmodel")),
            ],
        ),
        migrations.AddIndex(
            model_name="calibrationrecordmodel",
            index=models.Index(fields=["exercise", "mode"], name="api_calibra_exercis_33ba10_idx"),
        ),
        migrations.AddIndex(
            model_name="workoutsessionrecord",
            index=models.Index(fields=["session_type", "created_at"], name="api_workout_session_d9cb16_idx"),
        ),
        migrations.AddIndex(
            model_name="workoutsessionrecord",
            index=models.Index(fields=["exercise", "created_at"], name="api_workout_exercis_a46201_idx"),
        ),
    ]
