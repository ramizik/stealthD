"""
Speed and Distance Calculator using transformed positions from tracks.

This calculator works with tracks that have 'position_transformed' field and
calculates speed and distance using windowed frame batches.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import cv2
import numpy as np

from utils import get_foot_position, measure_distance


class SpeedAndDistance_Estimator:
    """
    Calculate speed and distance for tracked objects using transformed positions.

    Works with tracks that have 'position_transformed' field and calculates
    speed/distance using windowed frame batches.
    """

    def __init__(self, frame_window=10, frame_rate=24, min_distance_threshold=0.05, smoothing_window=3,
                 use_position_adjusted_fallback=True, max_speed_kmh=50.0, pixel_to_meter_ratio=0.008):
        """
        Initialize the SpeedAndDistance_Estimator.

        Args:
            frame_window: Number of frames to use for speed calculation window (default: 10, increased for better accuracy)
            frame_rate: Frame rate of the video in fps (default: 24)
            min_distance_threshold: Minimum distance in meters to consider as movement (filters noise, default: 0.05m)
            smoothing_window: Number of frames to use for speed smoothing (default: 3)
            use_position_adjusted_fallback: If True, use position_adjusted when position_transformed is missing (default: True, with proper conversion)
            max_speed_kmh: Maximum reasonable speed in km/h to cap calculations (default: 50.0 km/h for soccer players)
            pixel_to_meter_ratio: Conversion factor from pixels to meters when using position_adjusted (default: 0.008, estimated from typical field dimensions)
        """
        self.frame_window = frame_window
        self.frame_rate = frame_rate
        self.min_distance_threshold = min_distance_threshold  # Filter out noise below this distance
        self.smoothing_window = smoothing_window  # For smoothing speed values
        self.use_position_adjusted_fallback = use_position_adjusted_fallback
        self.max_speed_kmh = max_speed_kmh  # Cap unrealistic speeds
        self.pixel_to_meter_ratio = pixel_to_meter_ratio  # For pixel-to-meter conversion (conservative estimate)

    def add_speed_and_distance_to_tracks(self, tracks):
        """
        Add speed and distance calculations to tracks dictionary.

        The tracks structure should be:
        {
            'object_type': {
                frame_num: {
                    track_id: {
                        'position_transformed': [x, y] or None,
                        ...
                    }
                }
            }
        }

        This method adds 'speed' (km/h) and 'distance' (meters) to each track entry.

        Args:
            tracks: Dictionary of tracks with structure described above
        """
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            # Skip ball and referees
            if object_type == "ball" or object_type == "referees":
                continue

            # Skip metadata keys
            if object_type in ['player_classids', 'player_is_goalkeeper']:
                continue

            if not object_tracks:
                continue

            # Ensure object_tracks is a dictionary
            if not isinstance(object_tracks, dict):
                continue

            # Get frame indices
            frame_indices = sorted([f for f in object_tracks.keys() if isinstance(f, int)])
            if not frame_indices:
                continue

            # Initialize speed/distance to 0 for all players in all frames
            # This ensures all players get annotations even if they don't have position_transformed
            for frame_idx in frame_indices:
                if frame_idx not in object_tracks:
                    continue
                frame_tracks = object_tracks[frame_idx]
                if not isinstance(frame_tracks, dict):
                    continue
                for track_id, track_data in frame_tracks.items():
                    if track_id == -1:
                        continue
                    # Convert list format to dict if needed
                    if isinstance(track_data, list):
                        bbox = track_data
                        object_tracks[frame_idx][track_id] = {
                            'bbox': bbox,
                            'speed': 0.0,
                            'distance': 0.0
                        }
                    elif isinstance(track_data, dict):
                        # Ensure speed/distance fields exist
                        if 'speed' not in track_data:
                            track_data['speed'] = 0.0
                        if 'distance' not in track_data:
                            track_data['distance'] = 0.0
                        # Initialize total_distance if not exists
                        if object_type not in total_distance:
                            total_distance[object_type] = {}
                        if track_id not in total_distance[object_type]:
                            total_distance[object_type][track_id] = 0.0

            number_of_frames = len(frame_indices)

            # Store speed values for each frame to apply smoothing later
            speed_history = {}  # {track_id: [list of speeds per frame]}
            cumulative_distance_per_frame = {}  # {track_id: {frame_idx: cumulative_distance}}

            # First pass: calculate cumulative distance per frame
            for frame_num in range(number_of_frames):
                frame_idx = frame_indices[frame_num]

                if frame_idx not in object_tracks:
                    continue

                frame_tracks = object_tracks[frame_idx]
                if not isinstance(frame_tracks, dict):
                    continue

                for track_id, track_data in frame_tracks.items():
                    if track_id == -1:
                        continue
                    if not isinstance(track_data, dict):
                        continue

                    # Initialize cumulative distance for this track
                    if track_id not in cumulative_distance_per_frame:
                        cumulative_distance_per_frame[track_id] = {}

                    # Get cumulative distance from previous frame
                    prev_cumulative = 0.0
                    if frame_num > 0:
                        prev_frame_idx = frame_indices[frame_num - 1]
                        if prev_frame_idx in cumulative_distance_per_frame[track_id]:
                            prev_cumulative = cumulative_distance_per_frame[track_id][prev_frame_idx]

                    # Calculate distance from previous frame if available
                    # Use position_transformed if available, otherwise fall back to position_adjusted
                    frame_dist = 0.0
                    if frame_num > 0:
                        prev_frame_idx = frame_indices[frame_num - 1]
                        if prev_frame_idx in object_tracks:
                            prev_frame_tracks = object_tracks[prev_frame_idx]
                            if isinstance(prev_frame_tracks, dict) and track_id in prev_frame_tracks:
                                prev_track_data = prev_frame_tracks[track_id]
                                if isinstance(prev_track_data, dict):
                                    # Try position_transformed first, fall back to position_adjusted
                                    prev_position = prev_track_data.get('position_transformed', None)
                                    curr_position = track_data.get('position_transformed', None)

                                    # Fallback to position_adjusted if position_transformed is missing
                                    prev_using_pixel = False
                                    curr_using_pixel = False
                                    if prev_position is None and self.use_position_adjusted_fallback:
                                        prev_position = prev_track_data.get('position_adjusted', None)
                                        prev_using_pixel = True
                                    if curr_position is None and self.use_position_adjusted_fallback:
                                        curr_position = track_data.get('position_adjusted', None)
                                        curr_using_pixel = True

                                    if prev_position is not None and curr_position is not None:
                                        frame_dist = measure_distance(np.array(prev_position), np.array(curr_position))

                                        # Convert pixel distance to meters if using pixel coordinates
                                        # Only convert if at least one position is in pixels
                                        if prev_using_pixel or curr_using_pixel:
                                            frame_dist = frame_dist * self.pixel_to_meter_ratio

                                        # Only count if above noise threshold
                                        if frame_dist < self.min_distance_threshold:
                                            frame_dist = 0.0

                    # Update cumulative distance
                    cumulative_distance_per_frame[track_id][frame_idx] = prev_cumulative + frame_dist

                    # Also update total_distance for backward compatibility
                    if object_type not in total_distance:
                        total_distance[object_type] = {}
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0.0
                    total_distance[object_type][track_id] = cumulative_distance_per_frame[track_id][frame_idx]

            # Second pass: calculate speed using sliding window approach
            for frame_num in range(number_of_frames):
                frame_idx = frame_indices[frame_num]

                if frame_idx not in object_tracks:
                    continue

                frame_tracks = object_tracks[frame_idx]
                if not isinstance(frame_tracks, dict):
                    continue

                for track_id, track_data in frame_tracks.items():
                    if track_id == -1:
                        continue

                    # Get current position
                    if isinstance(track_data, list):
                        continue
                    elif not isinstance(track_data, dict):
                        continue

                    # Try position_transformed first (meters), fall back to position_adjusted (pixels) if needed
                    current_position = track_data.get('position_transformed', None)
                    using_pixel_fallback = False

                    if current_position is None and self.use_position_adjusted_fallback:
                        current_position = track_data.get('position_adjusted', None)
                        using_pixel_fallback = True

                    if current_position is None:
                        # No position data available
                        if track_id not in speed_history:
                            speed_history[track_id] = []
                        speed_history[track_id].append(0.0)
                        continue

                    # Initialize speed history for this track
                    if track_id not in speed_history:
                        speed_history[track_id] = []

                    # Use sliding window: look back up to frame_window frames
                    window_start = max(0, frame_num - self.frame_window)

                    # Collect positions in the window
                    window_positions = []
                    window_frame_indices = []

                    for w_frame_num in range(window_start, frame_num + 1):
                        w_frame_idx = frame_indices[w_frame_num]
                        if w_frame_idx not in object_tracks:
                            continue
                        if track_id not in object_tracks[w_frame_idx]:
                            continue

                        w_track_data = object_tracks[w_frame_idx][track_id]
                        if not isinstance(w_track_data, dict):
                            continue

                        # Try position_transformed first, fall back to position_adjusted if needed
                        w_position = w_track_data.get('position_transformed', None)
                        w_using_pixel = False

                        if w_position is None and self.use_position_adjusted_fallback:
                            w_position = w_track_data.get('position_adjusted', None)
                            w_using_pixel = True

                        if w_position is not None:
                            window_positions.append((np.array(w_position), w_using_pixel))
                            window_frame_indices.append(w_frame_idx)

                    # Need at least 2 points to calculate speed
                    if len(window_positions) < 2:
                        speed_history[track_id].append(0.0)
                        continue

                    # Calculate total distance traveled in the window (filtering noise)
                    total_distance_in_window = 0.0
                    for i in range(1, len(window_positions)):
                        pos1, pixel1 = window_positions[i-1]
                        pos2, pixel2 = window_positions[i]

                        dist = measure_distance(pos1, pos2)

                        # Convert pixel distance to meters if using pixel coordinates
                        if pixel1 or pixel2:
                            dist = dist * self.pixel_to_meter_ratio

                        # Filter out noise: only count distances above threshold
                        if dist >= self.min_distance_threshold:
                            total_distance_in_window += dist

                    # Calculate time elapsed in the window
                    time_elapsed = (window_frame_indices[-1] - window_frame_indices[0]) / self.frame_rate

                    if time_elapsed <= 0:
                        speed_history[track_id].append(0.0)
                        continue

                    # Calculate average speed over the window
                    speed_meters_per_second = total_distance_in_window / time_elapsed if time_elapsed > 0 else 0.0
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Cap speed at maximum reasonable value (sanity check)
                    if speed_km_per_hour > self.max_speed_kmh:
                        speed_km_per_hour = self.max_speed_kmh

                    # Store speed for this frame
                    speed_history[track_id].append(speed_km_per_hour)

            # Apply smoothing to speed values and update tracks
            for track_id, speeds in speed_history.items():
                if not speeds:
                    continue

                # Apply moving average smoothing with additional sanity checks
                smoothed_speeds = []
                for i in range(len(speeds)):
                    start_idx = max(0, i - self.smoothing_window // 2)
                    end_idx = min(len(speeds), i + self.smoothing_window // 2 + 1)
                    window_speeds = speeds[start_idx:end_idx]

                    # Filter out unrealistic speeds before averaging
                    valid_speeds = [s for s in window_speeds if s <= self.max_speed_kmh]

                    if valid_speeds:
                        smoothed_speed = np.mean(valid_speeds)
                        # Additional cap after smoothing
                        smoothed_speed = min(smoothed_speed, self.max_speed_kmh)
                    else:
                        smoothed_speed = 0.0

                    smoothed_speeds.append(smoothed_speed)

                # Update tracks with smoothed speeds and cumulative distance
                for frame_num, smoothed_speed in enumerate(smoothed_speeds):
                    if frame_num >= len(frame_indices):
                        break
                    frame_idx = frame_indices[frame_num]

                    if frame_idx not in object_tracks:
                        continue
                    if track_id not in object_tracks[frame_idx]:
                        continue

                    # Get cumulative distance for this frame
                    frame_distance = cumulative_distance_per_frame.get(track_id, {}).get(frame_idx, 0.0)

                    track_data = object_tracks[frame_idx][track_id]
                    if not isinstance(track_data, dict):
                        # Convert to dict
                        bbox = track_data if isinstance(track_data, list) else None
                        object_tracks[frame_idx][track_id] = {
                            'bbox': bbox,
                            'speed': smoothed_speed,
                            'distance': frame_distance
                        }
                    else:
                        track_data['speed'] = smoothed_speed
                        track_data['distance'] = frame_distance

    def draw_speed_and_distance_on_frame(self, frame, frame_idx, tracks):
        """
        Draw speed and distance annotations on a single video frame.

        Args:
            frame: Single video frame (numpy array)
            frame_idx: Frame index
            tracks: Dictionary of tracks with speed and distance information

        Returns:
            Annotated frame with speed and distance text
        """
        frame_copy = frame.copy()

        for object_type, object_tracks in tracks.items():
            # Only process player tracks (skip ball, referees, metadata)
            if object_type in ["ball", "referees", "referee", "player_classids", "player_is_goalkeeper"]:
                continue

            # Ensure object_tracks is a dictionary
            if not isinstance(object_tracks, dict):
                continue

            if frame_idx not in object_tracks:
                continue

            frame_tracks = object_tracks[frame_idx]

            # Ensure frame_tracks is a dictionary
            if not isinstance(frame_tracks, dict):
                continue

            for track_id, track_data in frame_tracks.items():
                # Skip invalid entries
                if track_id == -1:
                    continue

                # Extract bbox and speed/distance from track_data
                bbox = None
                speed = None
                distance = None

                # Handle both dict and list formats
                if isinstance(track_data, dict):
                    # Enhanced format: extract bbox, speed, distance
                    bbox = track_data.get('bbox', None)
                    speed = track_data.get('speed', None)
                    distance = track_data.get('distance', None)
                elif isinstance(track_data, list) and len(track_data) >= 4:
                    # Simple bbox format: use bbox, default speed/distance to 0
                    bbox = track_data
                    speed = 0.0
                    distance = 0.0
                else:
                    # Unknown format, skip
                    continue

                # Must have valid bbox to draw
                if bbox is None or len(bbox) < 4 or any(b is None for b in bbox):
                    continue

                # Use default values if speed/distance not calculated
                if speed is None:
                    speed = 0.0
                if distance is None:
                    distance = 0.0

                # Get foot position for text placement
                position = get_foot_position(bbox)
                position = list(position)
                position[1] += 40  # Offset below foot
                position = tuple(map(int, position))

                # Draw speed text
                cv2.putText(
                    frame_copy,
                    f"{speed:.2f} km/h",
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

                # Draw distance text below speed
                cv2.putText(
                    frame_copy,
                    f"{distance:.2f} m",
                    (position[0], position[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        return frame_copy

    def draw_speed_and_distance(self, frames, tracks):
        """
        Draw speed and distance annotations on video frames.

        Args:
            frames: List of video frames (numpy arrays)
            tracks: Dictionary of tracks with speed and distance information

        Returns:
            List of annotated frames
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            annotated_frame = self.draw_speed_and_distance_on_frame(frame, frame_num, tracks)
            output_frames.append(annotated_frame)

        return output_frames
