## Soccer Analysis System

Computer-vision pipeline for soccer: detects players/ball/referees, tracks entities, maps them to pitch coordinates, and generates metrics and context-aware insights.

### Run Locally
- Prereqs: Python 3.11+, Windows/macOS/Linux, optional NVIDIA GPU.
- Setup:
  - Create venv and install deps:
    - Windows (PowerShell):
      ```powershell
      python -m venv venv
      .\venv\Scripts\Activate
      pip install -r requirements.txt
      ```
    - macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      ```
  - Download models (recommended via HuggingFace Hub):
    ```bash
    # Object detection: players/ball/referees
    python -c "from huggingface_hub import hf_hub_download;import os,shutil;p=hf_hub_download(repo_id='Adit-jain/soccana',filename='best.pt');os.makedirs('Models/Trained/yolov11_sahi_1280/Model/weights',exist_ok=True);shutil.copy(p,'Models/Trained/yolov11_sahi_1280/Model/weights/best.pt');print('object model ready')"
    # Keypoint detection: 29 pitch keypoints
    python -c "from huggingface_hub import hf_hub_download;import os,shutil;p=hf_hub_download(repo_id='Adit-jain/Soccana_Keypoint',filename='best.pt');os.makedirs('Models/Trained/yolov11_keypoints_29/Model/weights',exist_ok=True);shutil.copy(p,'Models/Trained/yolov11_keypoints_29/Model/weights/best.pt');print('keypoint model ready')"
    ```
  - Place a video in `input_videos/` and set `test_video` in `constants.py` to its path.
- Run:
  ```bash
  python main.py
  ```
- Outputs:
  - Video: `output_videos/<name>_complete_analysis.mp4`
  - Data: `output_videos/<name>_analysis_data.json`

### Pipeline Highlights
- Adaptive Homography Mapper: Robust pitch mapping for \(x,y\) coordinates using dynamic keypoint confidence and fallback strategies.
- 29-Keypoint Field Detection: YOLOv11 pose model improves homography stability and tactical overlays.
- Camera Movement Estimator: Estimates pans/zooms/shakes to stabilize tracking and reduce false motion.
- Multi-Object Tracking: ByteTrack for persistent IDs; off-screen continuity heuristics.
- Two Pretrained Models:
  - Object detection (players/ball/referees): `Adit-jain/soccana`
  - Keypoint detection (29 points): `Adit-jain/Soccana_Keypoint`

### Tech Stack
- PyTorch: model inference and tensor ops
- Ultralytics / YOLOv8: detection and pose backbones
- NumPy: numerical arrays and vectorized math
- OpenCV (`cv2`): video I/O and image transforms
- supervision: detection utilities and visualization helpers
- Anthropic: LLM integration for context-aware suggestions

### Features
- Pass recognition
- Team classification
- Off-screen tracking continuity
- Goalkeeper recognition
- Speed detection
- Distance covered
- LLM integration for context-aware suggestions

### Current Limitations
- Processing time scales with video length; optimized for short clips (30s/1m/5m). Full matches can be slow.
- Accuracy can vary depending on footage:
  - Pass detection: may miss ambiguous touches or long aerials.
  - Persistent tracking: ID switches under heavy occlusion/camera cuts.
  - Keypoint detection: degraded under extreme occlusion or non-standard pitches.
  - Homography mapper: less stable on sparse keypoints or low-FOV angles.

### Roadmap
- Long video support (through segmenting, caching, incremental homography, resumable runs)
- Strategy and pattern analysis (pressing, overloads, progressive carries)
- Formation detection (per-phase shape estimation)
- Possession calculator (team/zone/time-weighted)

### Credits and Open Source
- Inspiration and code references:
  - `SoccerNet` ecosystem (datasets, protocols)
  - `abdullahtarek/football_analysis`
  - `roboflow` sports repositories
- Models and datasets:
  - Object detection model: `Adit-jain/Soccana_player_ball_detection` (HuggingFace `Adit-jain/soccana`)
  - Keypoint model: `Adit-jain/Soccana_Keypoint`
  - Pitch/keypoint data: Roboflow pitch detection datasets

### Troubleshooting
- Model not found: verify files in `Models/Trained/yolov11_sahi_1280/Model/weights` and `Models/Trained/yolov11_keypoints_29/Model/weights`.
- Video not found: ensure in `input_videos/` and `constants.py` `test_video` matches.
- OOM: try shorter clips, lower resolution, or close GPU-heavy apps.


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