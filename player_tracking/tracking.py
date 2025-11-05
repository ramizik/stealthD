import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import supervision as sv


class TrackerManager:
    """
    Manager class for player tracking functionality.
    Handles initialization and configuration of tracking models with advanced filtering.
    """

    def __init__(self, match_thresh=0.5, track_buffer=120, min_box_area=10, aspect_ratio_thresh=1.6):
        """
        Initialize the tracker with configurable parameters.

        Args:
            match_thresh (float): Matching threshold for tracking (default: 0.5)
            track_buffer (int): Number of frames to keep tracking buffer (default: 120)
            min_box_area (float): Minimum bounding box area to keep detection (default: 10)
            aspect_ratio_thresh (float): Maximum aspect ratio (h/w) for filtering vertical boxes (default: 1.6)
        """
        self.tracker = sv.ByteTrack()
        self.tracker.match_thresh = match_thresh
        self.tracker.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.aspect_ratio_thresh = aspect_ratio_thresh

    def get_tracker(self):
        """Get the configured tracker instance."""
        return self.tracker

    def update_player_detections(self, player_detections):
        """
        Update the player detections with the tracker.

        Args:
            player_detections: Detection results from YOLO

        Returns:
            Updated detections with tracker IDs
        """
        player_detections = self.tracker.update_with_detections(player_detections)
        return player_detections

    def process_tracking_for_frame(self, player_detections):
        """
        Process tracking for a single frame with quality filtering.

        Applies SoccerNet-style filtering:
        - Removes detections with area < min_box_area
        - Removes vertical boxes (aspect ratio > threshold)
        - Filters out low-quality detections

        Args:
            player_detections: Player detections for the frame

        Returns:
            Updated player detections with tracking information and quality filtering
        """
        if len(player_detections.xyxy) == 0:
            return player_detections

        # Apply pre-tracking filter to remove poor quality detections
        filtered_detections = self._filter_detections(player_detections)

        # Update tracker with filtered detections
        if len(filtered_detections.xyxy) > 0:
            filtered_detections = self.update_player_detections(filtered_detections)

        return filtered_detections

    def _filter_detections(self, detections):
        """
        Filter detections based on SoccerNet tracking criteria.

        Filters out:
        - Tiny boxes (area < min_box_area)
        - Vertical boxes (aspect_ratio > threshold) - often false positives

        Args:
            detections: sv.Detections object

        Returns:
            Filtered sv.Detections object
        """
        if len(detections.xyxy) == 0:
            return detections

        # Calculate box dimensions
        boxes = detections.xyxy
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        aspect_ratios = heights / (widths + 1e-6)  # h/w ratio

        # Filter criteria (relaxed from SoccerNet defaults)
        area_mask = areas > self.min_box_area
        aspect_mask = aspect_ratios <= self.aspect_ratio_thresh

        # Combined filter
        keep_mask = area_mask & aspect_mask

        # Debug: Print filtering stats on first few frames
        original_count = len(detections)
        filtered_count = keep_mask.sum()

        if not hasattr(self, '_filter_debug_printed'):
            if original_count > filtered_count:
                print(f"[TrackerManager] Filtering: {original_count} detections → {filtered_count} kept")
                print(f"  Area filter: {area_mask.sum()} passed (threshold: {self.min_box_area} px²)")
                print(f"  Aspect filter: {aspect_mask.sum()} passed (threshold: {self.aspect_ratio_thresh})")

                # Show examples of filtered detections
                rejected = ~keep_mask
                if rejected.any():
                    print(f"  Example rejected boxes:")
                    for i in np.where(rejected)[0][:3]:  # Show first 3
                        print(f"    Box {i}: area={areas[i]:.1f} px², aspect={aspect_ratios[i]:.2f}")
            self._filter_debug_printed = True

        # Apply filter
        if not keep_mask.any():
            # Return empty detections if all filtered out
            return sv.Detections.empty()

        return detections[keep_mask]