# DriveGuard AI
## Video-Based Advanced Driver Assistance System (ADAS) Prototype

DriveGuard AI is an experimental computer vision project designed to analyze road videos, detect and track road users, visualize their movement, and generate explainable, heuristic-based collision-risk indicators.

The system combines pretrained YOLOv8 object detection, ByteTrack multi-object tracking, OpenCV video processing, and a Streamlit dashboard to provide an end-to-end video analysis workflow.

This project demonstrates how object detection, tracking, temporal motion analysis, and interactive visualization can be integrated into a single Python application.

> **Safety Disclaimer**
>
> DriveGuard AI is an educational and research prototype, not a certified driver-assistance system. Its risk levels are experimental, uncalibrated estimates based on image-space heuristics. They are not collision probabilities or reliable real-world time-to-collision measurements. The system has not been validated for road use and must not be used to make driving decisions.

---

# 1. Project Overview

## 1.1 Problem Statement

Road videos contain multiple moving objects, including cars, buses, trucks, motorcycles, bicycles, and pedestrians. Understanding how these objects move relative to the camera can help demonstrate important computer vision concepts used in driver-assistance research.

However, object detection alone does not maintain object identities across video frames or provide an explanation of potential risk.

DriveGuard AI combines object detection, multi-object tracking, and rule-based motion analysis to explore these capabilities in a single application.

## 1.2 Project Objectives

- Detect road users in uploaded videos using a pretrained YOLOv8 model.
- Track detected objects across consecutive frames using ByteTrack.
- Display bounding boxes, object classes, detection confidence, and track IDs.
- Analyze image-space motion and bounding-box changes.
- Generate explainable heuristic risk indicators.
- Visualize original and annotated videos in a Streamlit dashboard.
- Export annotated videos and risk-event records.
- Inspect dataset integrity and evaluate supporting software components through automated tests.

## 1.3 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Ultralytics YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| OpenCV | Video reading, frame processing, annotation, and writing |
| Streamlit | Interactive web interface |
| Pandas | CSV processing and tabular analysis |
| NumPy | Numerical calculations |
| pytest | Unit and integration testing |
| Apple MPS / CPU | Model inference, depending on hardware support |

## 1.4 How the System Works

```text
Road Video
    |
    v
Video Frame Extraction
    |
    v
YOLOv8 Object Detection
    |
    v
ByteTrack Multi-Object Tracking
    |
    v
Track History and Motion Analysis
    |
    v
Heuristic Risk Engine
    |
    v
Annotated Video + Risk Event CSV
    |
    v
Streamlit Dashboard
    |
    v
Visual Results and Analysis Summary
```

---

# 2. Project Architecture and File Structure

The project is divided into application, processing, data, testing, and output components.

```text
DriveGuardAI/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train/
│   │   └── <video_id>.mp4
│   └── test/
│       └── <video_id>.mp4
│
├── models/
│
├── outputs/
│   └── annotated_videos/
│
├── results/
│
├── src/
│   ├── analyze_labels.py
│   ├── detect_video.py
│   ├── evaluate_collision.py
│   ├── inspect_dataset.py
│   ├── risk_engine.py
│   └── train_detector.py
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The exact contents of `data/`, `models/`, `outputs/`, `results/`, and `tests/` depend on the local project state and generated artifacts.

## 2.1 Understanding Each Component

### `app/dashboard.py`

This file contains the Streamlit application.

It provides the interface for:

- Uploading an MP4 video.
- Previewing the original video.
- Selecting a detector model and confidence threshold.
- Starting video analysis.
- Playing the annotated video.
- Reviewing tracking summaries and heuristic risk events.
- Downloading generated results.

### `src/detect_video.py`

This file implements the main video-processing pipeline.

It is responsible for:

- Opening the input video.
- Reading video frames.
- Running YOLO detection.
- Applying persistent ByteTrack tracking.
- Drawing annotations.
- Integrating the risk engine.
- Writing the annotated video and event records.

### `src/risk_engine.py`

This module evaluates tracked-object information and produces rule-based risk indicators.

It considers image position, relative image motion, bounding-box area changes, object class, and temporal confirmation.

### `src/inspect_dataset.py`

This utility checks dataset integrity, including CSV/video ID matching, file readability, video metadata, and reported data issues.

### `src/analyze_labels.py`

This script summarizes dataset label and timestamp fields.

### `src/evaluate_collision.py`

This script performs a timestamp-presence metadata sanity check. It is not a trained collision-prediction evaluation.

### `src/train_detector.py`

This script currently creates a training-plan summary. It does not train or fine-tune a detector or collision classifier.

### `tests/`

Contains automated tests for dataset inspection, risk-engine behavior, metric calculations, and video-processing integration.

### `data/`

Stores the local dataset and video files. Dataset files are not intended to be committed to the repository.

### `outputs/annotated_videos/`

Stores generated annotated videos and associated risk-event CSV files.

### `results/`

Stores dataset reports, label summaries, metadata checks, and other generated analysis artifacts.

### `models/`

Reserved for local model artifacts and checkpoints where applicable. The application may use pretrained YOLO weights downloaded or cached by Ultralytics.

---

# 3. Prerequisites

Before running DriveGuard AI, ensure that the following are available:

- Python compatible with the packages specified in `requirements.txt`.
- A terminal and VS Code or another code editor.
- OpenCV-compatible video support.
- An MP4 road video for analysis.
- Internet access for downloading pretrained model weights if they are not already available.

The project workspace has been verified using Python 3.13.

**Hardware support**

- macOS: Apple Silicon MPS acceleration may be available.
- Windows/Linux: CPU execution is supported; compatible acceleration depends on the installed PyTorch environment.
- CUDA: Not required.
- A GPU: Not mandatory.

---

# 4. Installation and Environment Setup

Follow these steps from the beginning when setting up the project on a new machine.

## Step 1: Open the project folder

Open the `DriveGuardAI` folder in VS Code.

Open the integrated terminal using:

**Terminal → New Terminal**

Navigate to the project root:

```bash
cd /path/to/DriveGuardAI
```

Replace the example path with your actual project location.

Confirm that you are in the correct directory:

```bash
pwd
```

List the project files:

```bash
ls
```

You should see files and folders such as `app`, `src`, `tests`, and `requirements.txt`.

## Step 2: Create a virtual environment

A virtual environment isolates project dependencies from other Python projects.

For macOS/Linux:

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

For Windows Command Prompt:

```bat
python -m venv venv
venv\Scripts\activate.bat
```

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

After activation, the terminal usually displays `(venv)`.

## Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

`pip` is Python's package installer.

## Step 4: Install project dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the packages listed in the project's dependency file.

If installation fails, inspect the Python version and the package versions specified in `requirements.txt` before changing dependencies.

## Step 5: Verify the environment

Check Python:

```bash
python --version
```

Check whether the core packages import successfully:

```bash
python -c "import cv2, torch, streamlit, ultralytics; print('Core dependencies imported successfully')"
```

If this command succeeds, the core application dependencies can be imported.

To check Apple Silicon acceleration:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

A result of `True` means PyTorch reports that MPS is available. Otherwise, the application can use CPU execution.

---

# 5. Dataset Preparation

DriveGuard AI includes utilities for inspecting a local Nexar road-video dataset.

The application can also analyze a separate compatible MP4 video uploaded through Streamlit.

## 5.1 Expected dataset layout

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── train/
│   ├── 00000.mp4
│   ├── 00001.mp4
│   └── ...
└── test/
    ├── <video_id>.mp4
    └── ...
```

Use the actual filenames supplied with your dataset. The example filenames above are illustrative.

## 5.2 Expected CSV fields

The training CSV is expected to contain:

| Field | Description |
|---|---|
| `id` | Video identifier |
| `time_of_event` | Dataset-provided event timestamp field |
| `time_of_alert` | Dataset-provided alert timestamp field |
| `target` | Video-level target label |

The test CSV is expected to contain the `id` field.

The project reads video IDs as strings to preserve leading zeros when matching CSV records to video filenames.

## 5.3 Inspect dataset integrity

Run:

```bash
python src/inspect_dataset.py --project-root .
```

This generates:

```text
results/
├── dataset_report.json
└── dataset_report.txt
```

The reports summarize CSV/video matching, video metadata, readability, label counts, timestamps, and detected issues.

Review the text report before proceeding if your dataset contains missing files or unreadable videos.

## 5.4 Summarize labels

Run:

```bash
python src/analyze_labels.py \
  --csv data/train.csv \
  --output results/label_summary.json
```

This creates a summary of the available label and timestamp fields.

## 5.5 Important dataset distinction

The Nexar `target` field is a video-level label, not an object-level bounding-box annotation.

The current project has not prepared a supervised object-detection dataset from Nexar video frames. It has not trained a collision classifier using these labels.

The existing video detector uses pretrained YOLO weights.

---

# 6. Running DriveGuard AI

Once the environment is configured, launch the Streamlit application.

From the project root:

```bash
python -m streamlit run app/dashboard.py --server.port 8505
```

Alternatively, if the virtual environment is not activated on macOS/Linux:

```bash
./venv/bin/python -m streamlit run app/dashboard.py --server.port 8505
```

Streamlit prints a local URL in the terminal, typically:

```text
http://localhost:8505
```

Open the URL in your browser.

Keep the terminal open while using the application. Press `Ctrl+C` in the terminal to stop the server.

If port 8505 is already occupied, start the application on another port:

```bash
python -m streamlit run app/dashboard.py --server.port 8506
```

---

# 7. Analyze a Road Video

The dashboard provides an end-to-end workflow for interactive video analysis.

## Step 1: Upload a video

Select an MP4 road video using the uploader.

The current Streamlit upload limit is 200 MB per file.

The application previews the original video before analysis.

## Step 2: Select the detector

Choose one of the available model variants:

- `yolov8n.pt`: Nano model, generally faster and lighter.
- `yolov8s.pt`: Small model, generally more computationally demanding.

The model is pretrained. Selecting a different model does not train it on the uploaded video.

## Step 3: Configure detection confidence

Adjust the confidence threshold.

The threshold controls which detections are retained for downstream processing. A lower threshold may retain more detections but can also introduce false positives.

## Step 4: Start analysis

Click **Analyze video**.

The application processes the video frame by frame.

For each frame, the pipeline performs object detection, tracking, annotation, and heuristic risk evaluation.

Processing time depends on video length, resolution, model selection, and available hardware.

## Step 5: View the annotated video

The output video includes annotations for detected objects, where available:

- Object class.
- Tracking ID.
- Detection confidence.
- Heuristic risk level.

The annotated video is displayed directly in the Streamlit page.

## Step 6: Review the results

The dashboard provides the available analysis summaries, including:

- Per-class track counts.
- The highest observed heuristic risk level.
- Per-track warning summaries.
- Risk-event history with estimated video timestamps.

## Step 7: Download outputs

Where available, download the annotated MP4 video and its associated risk-event CSV.

Generated outputs are stored under:

```text
outputs/annotated_videos/
```

The uploaded source video is written to a temporary file for processing and removed afterward.

**Interpretation:** If the application reports no elevated risk events, this means no elevated event was generated by the configured heuristic rules. It does not establish that the road scene is safe.

---

# 8. Command-Line Video Processing

The video pipeline can also be executed without opening the dashboard.

Run the following command from the project root:

```bash
python src/detect_video.py \
  --video data/train/00000.mp4 \
  --model yolov8n.pt \
  --output-dir outputs/annotated_videos
```

Replace the sample video path with a valid local MP4 file.

## 8.1 Available arguments

| Argument | Purpose | Default |
|---|---|---|
| `--video` | Input video path | Required |
| `--model` | YOLO model name or path | `yolov8n.pt` |
| `--output-dir` | Output directory | Defined by the script |
| `--conf` | Detection-confidence threshold | `0.25` |
| `--max-frames` | Maximum frames to process | No explicit frame limit |
| `--display` | Show processed frames in an OpenCV window | Disabled unless specified |

The output-directory default and exact generated filenames are determined by the current script implementation.

## 8.2 Run a short smoke test

To process only ten frames:

```bash
python src/detect_video.py \
  --video data/train/00000.mp4 \
  --model yolov8n.pt \
  --output-dir outputs/annotated_videos \
  --max-frames 10
```

This is useful for checking that the detector, video reader, tracker, and output writer work together.

Remove `--max-frames 10` to process the complete video.

## 8.3 Expected outputs

The pipeline is designed to produce:

- An annotated H.264 MP4 video.
- A risk-event CSV containing generated event records.

Output creation depends on successful processing of the input video.

Video timestamps are estimated from frame indices and the FPS reported by OpenCV. Variable-frame-rate videos and inaccurate metadata can make timestamps approximate.

---

# 9. Understanding the Risk Engine

The risk engine is implemented in:

```text
src/risk_engine.py
```

It receives information about tracked detections, including bounding boxes, object classes, track IDs, confidence values, and video timestamps.

It compares successive observations of each track and evaluates configured image-space cues.

## 9.1 Position and motion

The engine examines the object's position relative to an assumed ego-vehicle region and changes in its position across frames.

These measurements are expressed in image coordinates.

They do not directly provide real-world speed or distance.

## 9.2 Bounding-box area changes

A bounding box represents the image region occupied by a detected object.

If the box becomes larger over successive frames, the object may be approaching the camera. However, perspective, camera movement, object motion, and detector variation can also cause size changes.

Therefore, bounding-box growth alone cannot establish an actual collision trajectory.

## 9.3 Temporal confirmation

The engine applies configured thresholds and temporal confirmation to determine whether warning conditions persist.

This is intended to reduce reactions to isolated observations, but it does not guarantee that warnings are correct.

## 9.4 Risk levels

| Risk level | Color | Meaning |
|---|---|---|
| `NORMAL` | Green | No elevated heuristic condition was confirmed |
| `CAUTION` | Orange | A configured warning condition was detected |
| `HIGH_RISK` | Red | A higher-severity heuristic condition was detected |

Warnings may include explanatory text and an estimated TTC-like value when the observed track motion satisfies the engine's configured conditions.

These outputs are experimental heuristic indicators. They are not calibrated collision probabilities, verified time-to-collision measurements, or reliable statements about future events.

---

# 10. Dataset Metadata Evaluation

The project includes a timestamp-presence sanity check.

Run:

```bash
python src/evaluate_collision.py \
  --csv data/train.csv \
  --output results/collision_evaluation.json
```

The script writes its output to:

```text
results/collision_evaluation.json
```

The current check uses timestamp presence from the dataset labels and evaluates against the same training CSV.

It does not train a predictive model, use a held-out validation or test split, or establish collision-prediction accuracy.

**Do not report the resulting metrics as model performance.**

## Training-plan utility

Run:

```bash
python src/train_detector.py \
  --csv data/train.csv \
  --output results/training_plan.json
```

This generates a training-plan summary.

The current script is a scaffold; it does not train or fine-tune YOLO or a collision-prediction classifier.

---

# 11. Testing the Project

Run the automated test suite from the project root:

```bash
python -m pytest tests -q
```

The current tests cover:

- Dataset report generation and ID matching.
- Risk-engine behavior.
- Binary metric calculations.
- A short YOLO video-processing integration test.

The detector smoke test uses the local sample:

```text
data/train/00000.mp4
```

The first test run may download pretrained YOLO weights if they are not already available.

If the tests fail, review the traceback and verify the local dataset, installed dependencies, and model-weight availability.

Passing tests demonstrates the tested software behavior; it does not establish collision-prediction accuracy or automotive safety.

---

# 12. Implementation Status

| Feature | Status |
|---|---|
| Dataset CSV/video matching | Implemented |
| Video metadata and readability checks | Implemented |
| Pretrained YOLOv8 object detection | Implemented |
| Persistent ByteTrack tracking | Implemented |
| Bounding-box and track-ID annotation | Implemented |
| Heuristic risk indicators | Implemented |
| Streamlit upload and video playback | Implemented |
| Annotated MP4 and event CSV outputs | Implemented |
| Unit and integration tests | Implemented |
| Custom detector training or fine-tuning | Not implemented |
| Supervised Nexar bounding-box dataset preparation | Not implemented |
| Learned collision-prediction classifier | Not implemented |
| Held-out collision-prediction evaluation | Not implemented |
| Camera calibration and real-world distance estimation | Not implemented |
| Automotive safety certification | Not implemented |

---

# 13. Troubleshooting

### Problem: Streamlit port is already in use

Run the dashboard using another port:

```bash
python -m streamlit run app/dashboard.py --server.port 8506
```

### Problem: Python packages are missing

Activate the project virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

### Problem: The first run is slow

Ultralytics may need to download pretrained model weights. Subsequent runs may be faster if the weights are cached.

### Problem: No objects are detected

Try a clearer, well-lit video or lower the confidence threshold. Lowering the threshold may also increase false positives.

### Problem: No risk events are generated

Inspect the annotated video and track summaries. The configured rules may not trigger for the motion in the clip.

No generated alert does not establish that the scene is safe.

### Problem: The annotated video does not play

Confirm that the output file was created successfully. Try a current browser and verify that the generated video uses a browser-compatible codec.

### Problem: Dataset inspection reports missing files

Review:

```text
results/dataset_report.txt
```

Check the affected IDs and filenames, and confirm that CSV identifiers match the local video files.

### Problem: Video timestamps seem inaccurate

Inspect the video's FPS metadata and determine whether the source uses a variable frame rate.

---

# 14. Git and Repository Hygiene

Before pushing the project to GitHub, check `.gitignore`.

The repository should normally exclude:

- Virtual environments.
- Downloaded datasets and raw video files.
- Generated annotated videos.
- Local checkpoints and model weights.
- Temporary files, logs, and caches.
- Machine-specific configuration files.

Example `.gitignore` entries:

```gitignore
# Virtual environments
venv/
.venv/

# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Local datasets
data/

# Generated outputs and reports
outputs/
results/

# Local model artifacts
models/
*.pt
*.pth
*.onnx

# Environment variables and secrets
.env
.env.*

# Editor and operating-system files
.DS_Store
.vscode/
```

**Important:** This is an example, not a replacement for reviewing the existing `.gitignore`. If you want to version selected reports or model artifacts, adjust these patterns accordingly. Never commit private data, credentials, or files that you are not authorized to redistribute.

---

# 15. Current Limitations and Future Development

DriveGuard AI currently demonstrates a pretrained detection and tracking pipeline connected to an experimental heuristic risk engine.

The following are potential future extensions, not completed capabilities:

1. **Supervised dataset preparation:** Develop a properly defined training and evaluation dataset with appropriate annotations.
2. **Detector fine-tuning:** Fine-tune an object detector if the available annotations and project objectives justify it.
3. **Collision-prediction modelling:** Train a dedicated model using appropriate labels and a leakage-resistant evaluation protocol.
4. **Held-out evaluation:** Evaluate predictions on independent videos and report suitable metrics.
5. **Camera calibration:** Investigate methods for estimating physical geometry under documented assumptions.
6. **Ego-motion compensation:** Account for movement of the camera or host vehicle.
7. **Improved temporal modelling:** Investigate trajectory estimation and temporal filtering.
8. **Performance optimization:** Measure throughput, memory usage, and processing latency across supported hardware.
9. **Robustness testing:** Assess performance across different lighting, traffic conditions, camera viewpoints, and occlusions.

Any future collision-prediction model would require appropriate ground truth, independent evaluation, and extensive safety validation before its outputs could be considered for real-world use.

---

# 16. Final Execution Checklist

Use this checklist when setting up and demonstrating the project.

- [ ] Open the project root in VS Code.
- [ ] Create and activate the virtual environment.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Verify that core dependencies import successfully.
- [ ] Confirm that a valid MP4 road video is available.
- [ ] Run dataset inspection if using the Nexar dataset.
- [ ] Launch the Streamlit dashboard.
- [ ] Upload an MP4 video and configure the detector.
- [ ] Run the analysis and wait for processing to finish.
- [ ] Inspect the annotated video and heuristic risk summaries.
- [ ] Verify that the expected output files were created.
- [ ] Run the automated tests.
- [ ] Review `.gitignore` before pushing to GitHub.

---

# Conclusion

DriveGuard AI demonstrates an end-to-end computer vision workflow for road-video analysis, combining pretrained object detection, multi-object tracking, image-space motion heuristics, video annotation, and interactive visualization.

The project provides a practical foundation for exploring computer vision and experimental ADAS concepts while clearly separating implemented functionality from future research work.

**Project category:** Computer Vision · Object Detection · Multi-Object Tracking · Video Analytics · Experimental ADAS

**Primary language:** Python

**Interface:** Streamlit

**Safety status:** Educational prototype; not validated or certified for road use.
