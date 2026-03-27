from django.urls import path

from .views import (
    AnalyzeFormApiView,
    AskApiView,
    CalibrationSummaryApiView,
    RootApiView,
    SaveSessionApiView,
    SaveWorkoutApiView,
    SummaryApiView,
    WorkoutSessionDetailApiView,
    WorkoutSessionListApiView,
)


urlpatterns = [
    path("", RootApiView.as_view()),
    path("summary", SummaryApiView.as_view()),
    path("ask", AskApiView.as_view()),
    path("save_workout", SaveWorkoutApiView.as_view()),
    path("save_session", SaveSessionApiView.as_view()),
    path("analyze_form", AnalyzeFormApiView.as_view()),
    path("api/v1/", RootApiView.as_view()),
    path("api/v1/summary", SummaryApiView.as_view()),
    path("api/v1/ask", AskApiView.as_view()),
    path("api/v1/workout-sessions", WorkoutSessionListApiView.as_view()),
    path("api/v1/workout-sessions/<uuid:pk>", WorkoutSessionDetailApiView.as_view()),
    path("api/v1/calibrations/<str:exercise>", CalibrationSummaryApiView.as_view()),
    path("api/v1/form-analysis", AnalyzeFormApiView.as_view()),
]
