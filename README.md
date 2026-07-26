# FitCoachAR: Real-Time Adaptive Exercise Coaching via Pose Estimation and AR Feedback

**FitCoachAR** is a lightweight, real-time fitness coaching system that monitors exercises through 2D pose estimation, detects common form errors, and provides adaptive feedback through augmented-reality overlays and LLM-driven natural language coaching.

This project implements the concepts from:

- **Lecture 5: Human Kinematic Modeling** - Body-relative coordinate frames, forward kinematics, angular features
- **AIFit (CVPR 2021)** - Exercise modeling, repetition segmentation, active/passive features
- **Modern AR and LLM technologies** - Real-time visual feedback and natural language generation

---



## 🎯 Project Objectives

1. **Real-Time Pose Analysis**: Track push-ups, squats, and bicep curls using MediaPipe Pose
2. **Personalized Calibration**: Adapt thresholds using user's "best-form" repetitions
3. **Dynamic AR Visualization**: Arrows, colored joints, angle indicators for instant feedback
4. **LLM-Driven Coaching**: Natural language feedback generated from quantitative error analysis
5. **Session Summaries**: Comprehensive post-workout reports with recommendations

---



## 🏗️ Architecture



### Backend (Python/Django + DRF + Channels)

- `pose_backends/`: Pluggable pose-processing engines (MediaPipe 2D default, 3D-ready scaffold)
- `main.py`: ASGI entrypoint for the Django/Channels backend
- `api/models.py`: ORM models for calibrations, workout sessions, form analysis, and LLM logs
- `api/views.py`: DRF API endpoints for summaries, saved workouts/sessions, and analysis
- `api/admin.py`: Django admin configuration for inspecting workout/calibration data
- `kinematics.py`: Body-relative frames, angular feature extraction, forward kinematics
- `llm_feedback.py`: Template-based and API-driven natural language generation
- `filters.py`: Kalman filtering for landmark smoothing



### Frontend (React/Vite)

- `App.jsx`: Main application flow (calibration → workout → summary)
- `AROverlay.jsx`: Dynamic AR visualization with colored feedback
- `Avatar.jsx`: 3D skeleton rendering using Three.js

---



## 📚 Key Technical Concepts



### 1. Body-Relative Coordinate Frames (Lecture 5)

Implemented in `kinematics.py`:

```python
class BodyRelativeFrame:
    """
    - Origin: pelvis/hip center
    - X: Left-right (mediolateral)
    - Y: Vertical (superior-inferior)
    - Z: Front-back (anteroposterior)
    """
```

This normalization removes dependency on global orientation, making angle measurements consistent regardless of camera position.

### 2. Forward Kinematics & Angular Features

Extracts active (elbow, knee angles) and passive (spine, hip stability) features:

```python
features = AngularFeatureExtractor.extract_all_features(landmarks)
# Returns: elbow_angle, knee_angle, spine_angle, hip_tilt, etc.
```



### 3. Online Repetition Segmentation

State machine detects phase transitions (up → down → up):

```python
segmenter = RepetitionSegmenter(exercise_type='bicep_curls')
new_rep = segmenter.update(angle, thresholds)
```



### 4. LLM-Driven Feedback

Two-tier approach for low latency:

- **Real-time**: Template-based feedback (<100ms)
- **Summary**: Full LLM-generated session report

```python
feedback = llm_feedback.generate_feedback({
    "exercise": "bicep_curls",
    "errors": [{"joint": "right_elbow", "deviation_deg": 12}],
    "critic_level": 0.5
})
# Output: "Nice pace! Try curling a bit higher to finish each rep."
```



### 5. Dynamic AR Visualization

Real-time overlay features:

- ✅ **Green joints**: Correct form
- ❌ **Red joints**: Error detected
- 📐 **Angle arcs**: Visual angle indicators
- ➡️ **Yellow arrows**: Correction guidance

---



## 🚀 Installation & Setup



### Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam

### Backend Setup

```bash
cd fitcoachar/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
```

**Required packages** (`requirements.txt`):

```
fastapi
uvicorn[standard]
uvloop
opencv-python
mediapipe
numpy
pykalman
scipy
```



### Pose Backends

The backend now loads pose processors dynamically. Configure via the `POSE_BACKEND`
environment variable (defaults to `mediapipe_2d`):

```bash
export POSE_BACKEND=mediapipe_2d   # default 2.5D MediaPipe pipeline
# or
export POSE_BACKEND=mediapipe_3d   # MediaPipe world-landmark 3D pipeline
# Optional (requires extra deps/config):
export POSE_BACKEND=movenet_3d     # Calls external MoveNet microservice (see below)
export POSE_BACKEND=mmpose_poselifter  # MMPose PoseLifter (needs MMPOSE_CONFIG/MMPOSE_CHECKPOINT)
python manage.py runserver 0.0.0.0:8000
```

`GET /` reports the active backend plus all registered options so clients/frontends
can react accordingly.

> **Note:** `mediapipe_3d` shares the existing calibration logic and is ready to use.
> The `mmpose_poselifter` backend is still a scaffold and needs a full 2D detector +
> lifter integration before it produces results.



#### Calibration modes and critic controls

Every exercise now supports two runtime modes:

- **Common mode** – everyday training with the currently selected calibration.
- **Calibration mode** – review previous captures or record a new baseline.

When you record a new calibration the backend stores:

- Extended/contracted angles (or squat up/down)
- Per-joint deviation parameters (η)
- Critic thresholds (δ) for common and calibration modes
- Base64 snapshots of the captured poses for later review

Use the mode toggle in the UI to switch between common and calibration workflows, adjust
the critic level for each mode, and replay past calibrations (including the captured
snapshots and deviation metrics).

#### MoveNet (external microservice)

TensorFlow’s macOS build pins older `typing-extensions`/`numpy`, so we run MoveNet in a
separate environment and call it over HTTP.

1. **Create a TensorFlow environment**
  ```bash
   conda create -n movenet python=3.10
   conda activate movenet
   pip install tensorflow-macos==2.13.1 tensorflow-metal==1.0.0
   pip install numpy==1.24.3 typing-extensions<4.6 opencv-python==4.7.0.72 flask
  ```
2. **Download a TFLite MoveNet model** (e.g. MultiPose Lightning LiteRT) and note its path.
3. **Run the service**:
  ```bash
   python backend/services/movenet_service.py \
     --model /path/to/movenet_3d.tflite \
     --host 127.0.0.1 --port 8502
  ```
   The service exposes `POST /infer` and stays running in this environment.
4. **Back in the main FitCoachAR environment**, point the backend at the service:
  ```bash
   export POSE_BACKEND=movenet_3d
   export MOVENET_SERVICE_URL=http://127.0.0.1:8502/infer
  ```

python manage.py runserver 0.0.0.0:8000

For production-style realtime scaling, set `REDIS_URL` so Channels and Django cache can
share state through Redis:

```bash
export REDIS_URL=redis://127.0.0.1:6379/1
```



### Frontend Setup

```bash
cd fitcoachar/frontend
npm install
```

---



## ▶️ Running the Application



### 1. Start Backend Server

```bash
cd backend
source venv/bin/activate
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```



### Django Admin

Create a superuser and inspect persisted sessions/calibrations via the admin UI:

```bash
python manage.py createsuperuser
```

Then open `http://localhost:8000/admin/`.

### DRF API Examples

- `GET /api/v1/workout-sessions`
- `GET /api/v1/workout-sessions/<uuid>`
- `GET /api/v1/calibrations/bicep_curls`
- `POST /api/v1/summary`
- `POST /api/v1/form-analysis`



### 2. Start Frontend Dev Server

```bash
cd frontend
npm run dev
```



### 3. Open Application

Navigate to `http://localhost:5173`

---

## 📖 Usage Guide

### Step 1: Select Exercise

Choose between **Bicep Curls** or **Squats**

### Step 2: Calibration (Personalization)

- **For Bicep Curls**:
  1. Extend arm fully down → Record
  2. Curl to maximum height → Record
- **For Squats**:
  1. Stand straight → Record
  2. Descend to deepest squat → Record

This creates personalized thresholds adapted to your range of motion.

### Step 3: Workout

- Perform your exercise
- Watch for:
  - **Rep counter**: Automatically increments
  - **Form feedback**: "Keep elbow stable!" or "Go deeper!"
  - **LLM coaching**: Natural language tips
  - **AR overlay**: Visual error indicators



### Step 4: Summary

Review your session:

- Total reps completed
- Success rate
- Common mistakes identified
- Recommendations for next session

---

## 🎓 Educational Value

This project demonstrates:

1. **Human Kinematic Modeling**: Practical application of joint hierarchies, DoF, coordinate frames
2. **Real-Time Computer Vision**: Streaming pose estimation with <100ms latency
3. **State Machine Design**: Finite state automaton for repetition detection
4. **LLM Integration**: Prompt engineering for context-aware feedback
5. **Full-Stack Development**: React + FastAPI + WebSocket architecture

---

## 📝 References

1. **Fieraru et al.** (2021). *AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training*. CVPR 2021.
2. **Lecture 5: Human Kinematics**. McMaster University, Fall 2025.
3. **MediaPipe Pose**. Google Research. [https://google.github.io/mediapipe/solutions/pose](https://google.github.io/mediapipe/solutions/pose)
4. **Cao et al.** (2017). *Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields*. CVPR 2017.

---



## 👥 Contributors

**Course Project**: Mobile Data Analytics, McMaster University, Fall 2025

---



## 📄 License

This project is for educational purposes. See course guidelines for usage restrictions.

---



## 🤝 Acknowledgments

- **AIFit team** for the foundational methodology
- **MediaPipe team** for the pose estimation framework
- **Course instructors** for guidance on kinematic modeling

