import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

# Keypoint model paths - Updated to use new pitch detection model (32 keypoints)
keypoint_model_path = r"Models/Trained/pitch_detection/weights/best.pt"
keypoint_model_path = PROJECT_DIR / keypoint_model_path

# Dataset configuration
calibration_dataset_path = r"F:\Datasets\SoccerNet\Data\calibration"
test_images_path = calibration_dataset_path + r"\images\test"
test_labels_path = calibration_dataset_path + r"\labels\test"
test_video = PROJECT_DIR / r"input_videos\sample_1.mp4"
test_video_output = PROJECT_DIR / r"output_videos\sample_1_keypoints.mp4"

# Keypoint model configuration
NUM_KEYPOINTS = 32  # Updated from 29 to 32 for new pitch detection model
KEYPOINT_DIMENSIONS = 3  # (x, y, visibility)
CONFIDENCE_THRESHOLD = 0.5

# Soccer field keypoint names (32 keypoints) - Official SoccerNet layout
# Based on roboflow/sports repository SoccerPitchConfiguration
KEYPOINT_NAMES = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1",
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2",
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom",
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1",
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left",
    27: "center_circle_left",
    28: "center_circle_right",
    29: "big_rect_right_top_edge",        # Right penalty box top edge
    30: "sideline_bottom_right_corner",   # Bottom-right field corner (duplicate of 25)
    31: "center_circle_left_edge",        # Left edge of center circle on center line
}

# Keypoint connections for visualization - EXACT match to SoccerPitchConfiguration.edges
# Converted from 1-indexed to 0-indexed Python arrays
# Original edges from sports_compat.py SoccerPitchConfiguration:
# (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
# (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
# (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
# (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
# (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
# (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
KEYPOINT_CONNECTIONS = [
    # Horizontal edges (left side)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
    # Horizontal edges (center and right)
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
    (15, 16), (17, 18), (18, 19), (19, 20), (22, 23),
    # Horizontal edges (far right)
    (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    # Vertical/cross-field connections
    (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
    (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29)
]

# Field corner keypoint indices
FIELD_CORNERS = {
    'top_left': 0,
    'top_right': 16,
    'bottom_left': 9,
    'bottom_right': 25
}