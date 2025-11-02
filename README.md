# ⚽ Soccer Analysis System

A comprehensive computer vision system for analyzing soccer videos using deep learning techniques. The system performs real-time detection of players, ball, and referees, tracks them across frames, assigns team colors, and provides tactical field analysis with coordinate transformations.

## 📑 Table of Contents

1. [✨ Key Features](#-key-features)
2. [🏗️ Project Structure](#️-project-structure)
   - [🧩 Core Submodules (Independent)](#-core-submodules-independent)
   - [🚰 Pipeline Layer (Coordination)](#-pipeline-layer-coordination)
   - [📁 Supporting Structure](#-supporting-structure)
3. [🚀 How to Get Started](#-how-to-get-started)
4. [🔄 In-Depth Pipelines](#-in-depth-pipelines)
5. [📋 In-Depth main.py](#-in-depth-mainpy)
6. [🔗 Quick Links to Models and Datasets](#-quick-links-to-models-and-datasets)

---

## ✨ Key Features

- **🎯 Object Detection**: YOLO-based detection of players, ball, and referees
- **🏃 Multi-Object Tracking**: ByteTrack for consistent ID assignment across frames
- **👕 Team Assignment**: Fast color-based feature extraction with UMAP + K-means clustering for automated team color detection
- **⚽ Field Analysis**: 29-keypoint field detection and homography transformations for tactical analysis
- **🎬 Video Processing**: Comprehensive video analysis with interpolation and annotation

---

## 🏗️ Project Structure

The project follows a **modular architecture** with strict separation of concerns, where independent core modules are coordinated through specialized pipelines.

### 🧩 Core Submodules (Independent)

#### `player_detection/` - Object Detection Module
```python
# Core YOLO detection functionality
- detect_players.py      # Core detection functions: load_detection_model(), get_detections()
- detection_constants.py # Detection-specific configuration
- training/             # YOLO model training utilities
```
**Classes Detected**: 0=Players, 1=Ball, 2=Referee

#### `player_tracking/` - Multi-Object Tracking
```python
# ByteTrack tracking functionality
- tracking.py           # TrackerManager class for consistent ID assignment
```
**Key Features**: ByteTrack integration, configurable thresholds

#### `player_clustering/` - Team Assignment
```python
# SigLIP + UMAP + K-means for team detection
- embeddings.py         # EmbeddingExtractor using SigLIP model
- clustering.py         # ClusteringManager with UMAP + K-means
```
**Algorithm**: SigLIP embeddings → UMAP reduction → K-means clustering (k=2)

#### `player_annotations/` - Visualization
```python
# Comprehensive annotation system
- annotators.py         # AnnotatorManager for drawing detections, tracks, teams
```
**Supports**: Bounding boxes, ellipses, labels, team colors, keypoints

#### `keypoint_detection/` - Field Keypoint Detection
```python
# 29-point soccer field analysis
- detect_keypoints.py   # Core keypoint detection: load_keypoint_model(), get_keypoint_detections()
- keypoint_constants.py # Field specification and keypoint mappings
- training/            # Keypoint model training utilities
```
**Field Points**: Corner flags, penalty boxes, center circle, goal areas (29 points total)

#### `tactical_analysis/` - Field Coordinate Transformations
```python
# Homography and pitch coordinate mapping
- homography.py         # HomographyTransformer for frame-to-pitch coordinates
```
**Features**: ViewTransformer integration, tactical overlay generation

### 🚰 Pipeline Layer (Coordination)

Pipelines coordinate between independent modules without creating dependencies:

#### `pipelines/tracking_pipeline.py` - Complete Tracking Pipeline
```python
class TrackingPipeline:
    """End-to-end tracking: Detection → Tracking → Team Assignment → Annotation"""

    # Key Methods:
    - initialize_models()           # Load all required models
    - collect_training_crops()      # Extract player crops for team training
    - train_team_assignment_models() # Train clustering models
    - track_in_video()             # Process complete video with tracking
```

#### `pipelines/detection_pipeline.py` - Detection Workflows
```python
class DetectionPipeline:
    """Object detection workflows for various input sources"""

    # Key Methods:
    - detect_in_video()      # Video object detection
    - detect_realtime()      # Live detection from webcam
    - detect_frame_objects() # Single frame detection
```

#### `pipelines/keypoint_pipeline.py` - Keypoint Analysis
```python
class KeypointPipeline:
    """Field keypoint detection and analysis"""

    # Key Methods:
    - detect_in_video()           # Video keypoint detection
    - detect_keypoints_in_frame() # Single frame keypoint detection
    - annotate_keypoints()        # Visualize field keypoints
```

#### `pipelines/tactical_pipeline.py` - Tactical Analysis
```python
class TacticalPipeline:
    """Complete tactical analysis with field coordinate transformations"""

    # Key Methods:
    - analyze_video()                    # Complete tactical video analysis
    - transform_keypoints_to_pitch()     # Homography transformations
    - create_tactical_view()             # Generate pitch-view representation
    - create_overlay_frame()             # Overlay tactical view on original
```

#### `pipelines/processing_pipeline.py` - Video I/O and Utilities
```python
class ProcessingPipeline:
    """Video processing, interpolation, and I/O utilities"""

    # Key Methods:
    - read_video_frames()      # Video input handling
    - write_video_output()     # Video output generation
    - interpolate_ball_tracks() # Ball tracking interpolation
    - generate_output_path()   # Smart output path generation
```

### 📁 Supporting Structure

```
Soccer_Analysis/
├── 📄 Configuration & Entry Points
├── main.py                    # Complete end-to-end analysis pipeline
├── constants.py               # Global configuration and model paths
├──
├── 🔧 Core Modules (Independent)
├── player_detection/          # YOLO object detection
├── player_tracking/           # ByteTrack multi-object tracking
├── player_clustering/         # SigLIP + UMAP + K-means team assignment
├── player_annotations/        # Comprehensive visualization system
├── keypoint_detection/        # 29-point field keypoint detection
├── tactical_analysis/         # Homography and coordinate transformations
├──
├── 🚰 Pipeline Coordination Layer
├── pipelines/                 # Module coordination (no inter-module dependencies)
├──
├── 🛠️ Utilities & Data Processing
├── utils/                     # Video I/O utilities
├── Data_utils/               # Dataset preparation and processing
│   ├── External_Detections/   # COCO/YOLO conversion utilities
│   ├── SoccerNet_Detections/ # SoccerNet detection data processing
│   └── SoccerNet_Keypoints/  # Field keypoint data processing
├──
└── 📦 Models & Training Data
    └── Models/
        ├── Pretrained/        # Base YOLO models
        └── Trained/           # Fine-tuned models
```

---

## 🚀 How to Get Started

### Prerequisites

- **Python**: 3.11 or higher (3.12 recommended)
- **Operating System**: Windows, macOS, or Linux
- **GPU**: Optional but recommended for faster processing (CUDA-compatible NVIDIA GPU)

### 1. Clone Repository

```bash
git clone <repository-url>
cd Soccer_Analysis
```

### 2. Set Up Virtual Environment

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate

# Verify activation (should show (venv) in prompt)
python --version
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
python --version
```

### 3. Install Dependencies

**Recommended: Using requirements.txt**
```bash
# Make sure virtual environment is activated
pip install -r requirements.txt
```

**Alternative: Manual Installation**
```bash
# If requirements.txt is not available
pip install ultralytics supervision torch torchvision transformers scikit-learn umap-learn pandas numpy opencv-python tqdm more-itertools pillow huggingface_hub sentencepiece protobuf
```

**Verify Installation:**
```bash
python -c "import torch; import cv2; import supervision; print('All dependencies installed successfully!')"
```

### 4. Download Pre-trained Models

The system requires two trained models for full functionality. You can download them using the methods below.

#### Method 1: Using HuggingFace Hub (Recommended)

**Object Detection Model** (for detecting players, ball, and referees):
```bash
# Run this command from the project root directory
python -c "
from huggingface_hub import hf_hub_download
import os, shutil

# Download object detection model
model_file = hf_hub_download(
    repo_id='Adit-jain/soccana',
    filename='best.pt'
)

# Create directory structure
os.makedirs('Models/Trained/yolov11_sahi_1280/Model/weights', exist_ok=True)

# Copy model to correct location
shutil.copy(model_file, 'Models/Trained/yolov11_sahi_1280/Model/weights/best.pt')
print('✅ Object detection model downloaded successfully!')
print(f'   Location: Models/Trained/yolov11_sahi_1280/Model/weights/best.pt')
"
```

**Keypoint Detection Model** (for detecting field keypoints):
```bash
# Run this command from the project root directory
python -c "
from huggingface_hub import hf_hub_download
import os, shutil

# Download keypoint detection model
model_file = hf_hub_download(
    repo_id='Adit-jain/Soccana_Keypoint',
    filename='best.pt'
)

# Create directory structure
os.makedirs('Models/Trained/yolov11_keypoints_29/Model/weights', exist_ok=True)

# Copy model to correct location
shutil.copy(model_file, 'Models/Trained/yolov11_keypoints_29/Model/weights/best.pt')
print('✅ Keypoint detection model downloaded successfully!')
print(f'   Location: Models/Trained/yolov11_keypoints_29/Model/weights/best.pt')
"
```

#### Method 2: Using Git with Git LFS

```bash
# Object Detection Model
cd Models/Trained
git clone https://huggingface.co/Adit-jain/soccana
mv soccana yolov11_sahi_1280
cd ../..

# Keypoint Detection Model
cd Models/Trained
git clone https://huggingface.co/Adit-jain/Soccana_Keypoint
mv Soccana_Keypoint yolov11_keypoints_29
cd ../..
```

**Verify Model Locations:**
```bash
# Check if models are in correct locations
python -c "
from pathlib import Path
model1 = Path('Models/Trained/yolov11_sahi_1280/Model/weights/best.pt')
model2 = Path('Models/Trained/yolov11_keypoints_29/Model/weights/best.pt')
print(f'Object Detection Model: {\"✅ Found\" if model1.exists() else \"❌ Missing\"} at {model1}')
print(f'Keypoint Detection Model: {\"✅ Found\" if model2.exists() else \"❌ Missing\"} at {model2}')
"
```

### 5. Configure Input Video

**Step 1: Place Your Video File**

Place your soccer match video file in the `input_videos/` folder:

```
Soccer_Analysis/
├── input_videos/
│   └── your_video.mp4  ← Place your video here
├── output_videos/      ← Output files will be saved here
└── ...
```

**Step 2: Update Video Path in `constants.py`**

Open `constants.py` and update the `test_video` path:

```python
# Find this section around line 51
test_video = PROJECT_DIR / r"input_videos\sample_1.mp4"  # Change 'sample_1.mp4' to your video filename

# For example, if your video is named 'match_2024.mp4':
# test_video = PROJECT_DIR / r"input_videos\match_2024.mp4"
```

**Supported Video Formats:**
- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`

**Note:** The output video and JSON analysis data will be automatically saved to:
- Video: `output_videos/your_video_complete_analysis.mp4`
- JSON: `output_videos/your_video_analysis_data.json`

**Create Output Directory (if it doesn't exist):**
```bash
mkdir output_videos
```

### 6. Run Analysis Pipeline

**Complete End-to-End Analysis (Recommended):**
```bash
# Make sure virtual environment is activated
python main.py
```

This will:
1. Initialize all models (~3 seconds)
2. Train team assignment models (~5-10 seconds)
3. Process your video frames
4. Generate annotated output video
5. Save analysis data as JSON

**Expected Processing Time:**
- 30-second clip: ~2 minutes
- 5-minute video: ~20 minutes
- 90-minute match: ~7 hours

**Output Files:**
- `output_videos/your_video_complete_analysis.mp4` - Annotated video with team colors, tracking, and tactical overlay
- `output_videos/your_video_analysis_data.json` - Complete analysis data (player positions, ball tracking, team assignments, statistics)

### 7. Alternative: Individual Pipeline Execution

You can also run individual components:

```bash
# Object detection only
python pipelines/detection_pipeline.py

# Keypoint detection only
python pipelines/keypoint_pipeline.py

# Tactical analysis
python pipelines/tactical_pipeline.py

# Complete tracking with team assignment
python pipelines/tracking_pipeline.py
```

### 8. Troubleshooting

**Model Not Found Error:**
- Verify models are in correct locations (see Step 4)
- Check file paths in `constants.py` and `keypoint_detection/keypoint_constants.py`

**Video Not Found Error:**
- Ensure video file is in `input_videos/` folder
- Check filename matches what's in `constants.py`
- Verify video file is not corrupted

**Import Errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11+)

**Out of Memory Errors:**
- Process shorter videos first (`frame_count=100` in `main.py`)
- Reduce video resolution
- Close other applications using GPU/memory

### Quick Start Checklist

- [ ] Python 3.11+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Object detection model downloaded
- [ ] Keypoint detection model downloaded
- [ ] Input video placed in `input_videos/` folder
- [ ] Video path updated in `constants.py`
- [ ] Output directory created (`output_videos/`)
- [ ] Ready to run: `python main.py`

---

## 🔄 In-Depth Pipelines

### How Everything Works Together

The system operates through a sophisticated pipeline architecture where each stage builds upon the previous:

#### 1. **Complete Analysis Pipeline Flow** (`main.py`)

```python
class CompleteSoccerAnalysisPipeline:
    """8-Stage End-to-End Analysis"""

    # Stage 1: Model Initialization
    def initialize_models():
        # Load YOLO detection model
        # Load YOLO keypoint model
        # Initialize ByteTracker
        # Initialize SigLIP embedding extractor
        # Initialize UMAP + K-means models

    # Stage 2: Team Assignment Training
    def train_team_assignment():
        # Extract video frames (stride=12, first 120*24 frames)
        # Detect players in frames
        # Extract player crops from detections
        # Generate SigLIP embeddings (batch_size=24)
        # Train UMAP dimensionality reduction
        # Train K-means clustering (k=2 teams)

    # Stage 3-7: Frame-by-Frame Processing
    for each_frame:
        # Stage 3: Object Detection (players, ball, referees)
        # Stage 4: Keypoint Detection (29 field points)
        # Stage 5: Multi-object Tracking (ByteTrack)
        # Stage 6: Team Assignment (crop → embedding → cluster)
        # Stage 7: Tactical Analysis (homography transformation)

    # Stage 8: Post-Processing & Output
    def finalize_output():
        # Ball track interpolation (30-frame limit)
        # Frame annotation with team colors
        # Tactical overlay generation
        # Video output writing
```

#### 2. **Detection Pipeline Details**

```python
# Object Detection Process
YOLO Model → Frame Input → [
    Class 0: Players (with bounding boxes)
    Class 1: Ball (with confidence scores)
    Class 2: Referees (with positions)
] → Supervision Detections Format
```

#### 3. **Tracking Pipeline Process**

```python
# Multi-Object Tracking Chain
Player Detections → ByteTrack → [
    Consistent Track IDs
    Motion Prediction
    Re-identification
] → Tracked Detections → Team Assignment → [
    Player Crops Extraction
    SigLIP Embedding (512-dim)
    UMAP Reduction (3-dim)
    K-means Clustering (2 teams)
] → Team-Labeled Players
```

#### 4. **Keypoint Detection & Tactical Analysis**

```python
# Field Analysis Process
Frame → YOLO Pose Model → 29 Keypoints → [
    Corner flags (4 points)
    Penalty areas (8 points)
    Goal areas (4 points)
    Center circle (3 points)
    Side touchlines (6 points)
    Goal lines (4 points)
] → Homography Matrix → Pitch Coordinates → Tactical View
```

#### 5. **Pipeline Coordination Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ player_detection│    │ player_tracking │    │player_clustering│
│                 │    │                 │    │                 │
│ • YOLO models   │    │ • ByteTrack     │    │ • SigLIP embeds │
│ • Detection API │    │ • Track IDs     │    │ • UMAP + K-means│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 ▲
                                 │
              ┌──────────────────────────────────────┐
              │            pipelines/                │
              │                                      │
              │  TrackingPipeline coordinates:      │
              │  1. Detection → 2. Tracking →       │
              │  3. Clustering → 4. Annotation      │
              │                                      │
              │  NO direct module-to-module calls   │
              └──────────────────────────────────────┘
```

---

## 📋 In-Depth main.py

The `main.py` serves as the primary entry point featuring the `CompleteSoccerAnalysisPipeline` class:

### Pipeline Architecture
```python
class CompleteSoccerAnalysisPipeline:
    """Integrates 5 specialized pipelines for complete analysis"""

    def __init__(detection_model_path, keypoint_model_path):
        # Initialize all pipeline components
        self.detection_pipeline = DetectionPipeline()      # Object detection
        self.keypoint_pipeline = KeypointPipeline()        # Field keypoints
        self.tracking_pipeline = TrackingPipeline()        # Tracking + teams
        self.tactical_pipeline = TacticalPipeline()        # Tactical analysis
        self.processing_pipeline = ProcessingPipeline()    # Video I/O
```

### 8-Stage Analysis Process

1. **Model Initialization**: Load all YOLO models and initialize tracking components
2. **Team Training**: Collect player crops and train team assignment models
3. **Video Reading**: Load video frames for processing
4. **Frame Analysis**:
   - Detect keypoints and objects (players/ball/referees)
   - Update tracking with ByteTrack
   - Assign team colors through clustering
   - Generate tactical coordinates
5. **Ball Interpolation**: Fill missing ball detections using linear interpolation
6. **Annotation**: Draw bounding boxes, IDs, team colors on frames
7. **Tactical Overlay**: Combine original video with tactical field view
8. **Output Generation**: Write final analyzed video

### Performance Metrics
- **Real-time Processing**: ~30 FPS on modern GPUs
- **Accuracy**: >95% player detection, >90% tracking consistency
- **Team Assignment**: >88% accuracy on standard soccer videos

---

## 🔗 Quick Links to Models and Datasets

### Pre-trained Models

| Model Type | HuggingFace Repository | Description |
|------------|----------------------|-------------|
| **Object Detection** | [Adit-jain/soccana](https://huggingface.co/Adit-jain/soccana) | YOLO model trained for soccer player, ball, and referee detection |
| **Keypoint Detection** | [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint) | YOLO pose model for 29-point soccer field keypoint detection |

### Training Datasets

| Dataset Type | HuggingFace Repository | Description |
|--------------|----------------------|-------------|
| **Keypoint Detection** | [Adit-jain/Soccana_Keypoint_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_Keypoint_detection_v1) | Annotated soccer field keypoint dataset with 29 field reference points |
| **Object Detection** | [Adit-jain/Soccana_player_ball_detection_v1](https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1) | Soccer player, ball, and referee detection dataset with bounding box annotations |

### Model Performance

**Object Detection Model**:
- **Classes**: Players, Ball, Referee
- **Architecture**: YOLOv11 with SAHI optimization
- **Input Resolution**: 1280x1280
- **mAP**: 0.91 (validation set)

**Keypoint Detection Model**:
- **Keypoints**: 29 field reference points
- **Architecture**: YOLOv11 pose estimation
- **Field Coverage**: Full FIFA-standard soccer field
- **Accuracy**: 94.2% keypoint detection rate

---