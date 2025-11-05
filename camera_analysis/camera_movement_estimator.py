"""
Camera Movement Estimator using Optical Flow.

Uses Lucas-Kanade optical flow to detect camera pan/tilt movements
and compensate player tracking positions for improved accuracy.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import pickle
import cv2
import numpy as np
import os
import supervision as sv
from typing import Dict, List, Tuple
from utils import measure_distance, measure_xy_distance
from tqdm import tqdm


class CameraMovementEstimator:
    """
    Estimates camera movement using optical flow on static features.

    Detects camera pan/tilt by tracking features at video edges (where players
    are less likely to be) and compensates all track positions accordingly.
    """

    def __init__(self, frame):
        """
        Initialize camera movement estimator.

        Args:
            frame: First frame of the video (for dimensions and feature detection setup)
        """
        self.minimum_distance = 5  # Minimum pixel movement to consider camera motion

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Convert first frame to grayscale for feature detection
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = first_frame_grayscale.shape

        # Create dynamic camera movement masks based on video dimensions
        # Focus on edges where static features (crowd, ads) are more reliable
        mask_features = np.zeros_like(first_frame_grayscale)

        # Left edge mask (20 pixels or 2% of width, whichever is smaller)
        left_edge = min(20, int(frame_width * 0.02))
        mask_features[:, 0:left_edge] = 1

        # Right edge mask (150 pixels or 10% of width, whichever is smaller)
        right_edge_width = min(150, int(frame_width * 0.1))
        right_edge_start = max(frame_width - right_edge_width, int(frame_width * 0.85))
        mask_features[:, int(right_edge_start):frame_width] = 1

        # Good features to track parameters
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames: List, read_from_stub: bool = False,
                           stub_path: str = None) -> List[List[float]]:
        """
        Calculate camera movement for all frames using optical flow.

        Args:
            frames: List of video frames
            read_from_stub: If True, try to load cached results
            stub_path: Path to pickle file for caching

        Returns:
            List of [x_movement, y_movement] for each frame
        """
        # Try to load from cache
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
                print(f"  Loaded cached camera movement from: {stub_path}")
                return camera_movement

        print(f"  Analyzing camera movement across {len(frames)} frames...")

        # Initialize camera movement (no movement for first frame)
        camera_movement = [[0, 0]] * len(frames)

        # Get initial features from first frame
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process each frame
        for frame_num in range(1, len(frames)):
            # Show progress
            progress = (frame_num / len(frames)) * 100
            print(f"\r  Camera movement analysis: Frame {frame_num}/{len(frames)} ({progress:.1f}%)",
                  end='', flush=True)

            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # Find maximum movement among tracked features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )

            # Only update if movement exceeds threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Re-detect features for next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        print()  # New line after progress

        # Save to cache
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            print(f"  Camera movement cached to: {stub_path}")

        return camera_movement

    def get_camera_movement_streaming(self, video_path: str, total_frames: int,
                                      read_from_stub: bool = False,
                                      stub_path: str = None) -> List[List[float]]:
        """
        Calculate camera movement using streaming (memory-efficient).

        Args:
            video_path: Path to video file
            total_frames: Total number of frames to process
            read_from_stub: If True, try to load cached results
            stub_path: Path to pickle file for caching

        Returns:
            List of [x_movement, y_movement] for each frame
        """
        # Try to load from cache
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
                print(f"  Loaded cached camera movement from: {stub_path}")
                return camera_movement

        print(f"  Analyzing camera movement across {total_frames} frames (streaming)...")

        # Initialize camera movement (no movement for first frame)
        camera_movement = [[0, 0]] * total_frames

        # Use supervision's frame generator for memory efficiency
        frame_generator = sv.get_video_frames_generator(video_path, end=total_frames)

        # Get first frame and initial features
        first_frame = next(frame_generator)
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Process remaining frames
        update_interval = max(1, total_frames // 50)  # Update 50 times max
        for frame_num, frame in enumerate(tqdm(frame_generator, total=total_frames-1,
                                               desc="Camera movement",
                                               mininterval=2.0, miniters=update_interval,
                                               ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'),
                                        start=1):  # Start from frame 1

            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # Find maximum movement among tracked features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            if new_features is not None and old_features is not None:
                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = measure_distance(new_features_point, old_features_point)

                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(
                            old_features_point, new_features_point
                        )

            # Only update if movement exceeds threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Re-detect features for next frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        # Save to cache
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            print(f"  Camera movement cached to: {stub_path}")

        return camera_movement

    def compensate_track_positions(self, tracks: Dict, camera_movement: List[List[float]]) -> Dict:
        """
        Compensate all track positions for camera movement.

        Adjusts bounding boxes to account for camera pan/tilt, making positions
        relative to the field rather than the camera frame.

        Args:
            tracks: Dictionary with structure:
                {
                    'player': {frame_idx: {player_id: [x1, y1, x2, y2]}},
                    'ball': {frame_idx: [x1, y1, x2, y2]},
                    'referee': {frame_idx: {ref_id: [x1, y1, x2, y2]}}
                }
            camera_movement: List of [x_movement, y_movement] per frame

        Returns:
            Modified tracks dictionary with compensated positions
        """
        print(f"  Applying camera movement compensation to tracks...")

        # Compensate player tracks
        for frame_idx, frame_tracks in tracks['player'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            # Skip frames with no detections
            if -1 in frame_tracks:
                continue

            # Compensate each player's bbox
            for player_id, bbox in frame_tracks.items():
                if bbox is not None and all(b is not None for b in bbox):
                    x1, y1, x2, y2 = bbox
                    tracks['player'][frame_idx][player_id] = [
                        x1 - x_movement,
                        y1 - y_movement,
                        x2 - x_movement,
                        y2 - y_movement
                    ]

        # Compensate ball tracks
        for frame_idx, bbox in tracks['ball'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            if bbox is not None and all(b is not None for b in bbox):
                x1, y1, x2, y2 = bbox
                tracks['ball'][frame_idx] = [
                    x1 - x_movement,
                    y1 - y_movement,
                    x2 - x_movement,
                    y2 - y_movement
                ]

        # Compensate referee tracks
        for frame_idx, frame_tracks in tracks['referee'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            # Skip frames with no detections
            if -1 in frame_tracks:
                continue

            # Compensate each referee's bbox
            for ref_id, bbox in frame_tracks.items():
                if bbox is not None and all(b is not None for b in bbox):
                    x1, y1, x2, y2 = bbox
                    tracks['referee'][frame_idx][ref_id] = [
                        x1 - x_movement,
                        y1 - y_movement,
                        x2 - x_movement,
                        y2 - y_movement
                    ]

        print(f"  Camera movement compensation complete")

        return tracks

    def restore_camera_positions(self, tracks: Dict, camera_movement: List[List[float]]) -> Dict:
        """
        Restore original camera-relative positions by reversing compensation.

        Use this before annotation to draw on the original video frames.

        Args:
            tracks: Dictionary with compensated (field-relative) positions
            camera_movement: List of [x_movement, y_movement] per frame

        Returns:
            Modified tracks dictionary with camera-relative positions restored
        """
        # Restore player tracks
        for frame_idx, frame_tracks in tracks['player'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            # Skip frames with no detections
            if -1 in frame_tracks:
                continue

            # Restore each player's bbox (add camera movement back)
            for player_id, bbox in frame_tracks.items():
                if bbox is not None and all(b is not None for b in bbox):
                    x1, y1, x2, y2 = bbox
                    tracks['player'][frame_idx][player_id] = [
                        x1 + x_movement,
                        y1 + y_movement,
                        x2 + x_movement,
                        y2 + y_movement
                    ]

        # Restore ball tracks
        for frame_idx, bbox in tracks['ball'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            if bbox is not None and all(b is not None for b in bbox):
                x1, y1, x2, y2 = bbox
                tracks['ball'][frame_idx] = [
                    x1 + x_movement,
                    y1 + y_movement,
                    x2 + x_movement,
                    y2 + y_movement
                ]

        # Restore referee tracks
        for frame_idx, frame_tracks in tracks['referee'].items():
            if frame_idx >= len(camera_movement):
                continue

            x_movement, y_movement = camera_movement[frame_idx]

            # Skip frames with no detections
            if -1 in frame_tracks:
                continue

            # Restore each referee's bbox
            for ref_id, bbox in frame_tracks.items():
                if bbox is not None and all(b is not None for b in bbox):
                    x1, y1, x2, y2 = bbox
                    tracks['referee'][frame_idx][ref_id] = [
                        x1 + x_movement,
                        y1 + y_movement,
                        x2 + x_movement,
                        y2 + y_movement
                    ]

        return tracks
