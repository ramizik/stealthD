"""Ball Tracker for Soccer Analysis.

Handles anomalies in ball detection by filtering misidentified objects.
Uses temporal smoothing and centroid-based selection to ensure only the
most likely ball detection is tracked.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from collections import deque
from typing import Optional

import cv2
import numpy as np
import supervision as sv


class BallTracker:
    """
    Track soccer ball with anomaly filtering.

    The ball often gets misidentified due to its round shape and lack of
    distinct features. This tracker uses temporal smoothing and centroid-based
    selection to filter out errors and locate the actual ball.

    Assumptions:
    - Only one ball on the field
    - Ball movement is physically realistic (continuous, not teleporting)
    """

    def __init__(self, buffer_size: int = 10, max_distance: Optional[float] = None):
        """
        Initialize ball tracker.

        Args:
            buffer_size: Number of recent positions to store for centroid calculation
            max_distance: Maximum allowed distance from centroid (pixels). None = no limit
        """
        self.buffer = deque(maxlen=buffer_size)
        self.max_distance = max_distance

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Filter ball detections to select most likely ball.

        Process:
        1. Get center points of all ball detections
        2. Store in buffer for temporal averaging
        3. Calculate centroid from recent positions
        4. Select detection closest to centroid
        5. Optionally filter by max distance

        Args:
            detections: All ball detections for current frame

        Returns:
            Single detection (most likely ball) or empty if none valid
        """
        if len(detections) == 0:
            return detections

        # Get center coordinates of all ball detections
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        # Add to buffer for temporal smoothing
        self.buffer.append(xy)

        # Calculate centroid from recent positions (weighted average)
        if len(self.buffer) > 0:
            # Concatenate all positions in buffer
            all_positions = np.concatenate(list(self.buffer), axis=0)
            centroid = np.mean(all_positions, axis=0)
        else:
            # Fallback if buffer is empty
            centroid = np.mean(xy, axis=0)

        # Calculate distance from each detection to centroid
        distances = np.linalg.norm(xy - centroid, axis=1)

        # Find detection closest to centroid
        index = np.argmin(distances)

        # Optional: Filter by maximum distance
        if self.max_distance is not None and distances[index] > self.max_distance:
            # Ball too far from expected position - likely false detection
            return sv.Detections.empty()

        # Return single best detection
        return detections[[index]]


class BallAnnotator:
    """
    Visualize ball tracking with trail effect.

    Draws the ball position history with gradient colors and sizes,
    creating a motion trail effect similar to computer games.
    """

    def __init__(
        self,
        radius: int = 10,
        buffer_size: int = 5,
        thickness: int = 2
    ):
        """
        Initialize ball annotator.

        Args:
            radius: Maximum radius for most recent ball position
            buffer_size: Number of recent positions to show in trail
            thickness: Line thickness for circle drawing
        """
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolate radius from small (old) to large (recent).

        Args:
            i: Position index in buffer (0 = oldest)
            max_i: Buffer size

        Returns:
            Interpolated radius
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections
    ) -> np.ndarray:
        """
        Annotate frame with ball trail effect.

        Args:
            frame: Input frame
            detections: Ball detections (should be single detection)

        Returns:
            Annotated frame with ball trail
        """
        if len(detections) == 0:
            return frame

        # Get ball position (bottom center for ground contact point)
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)

        # Add to buffer
        self.buffer.append(xy)

        # Draw trail from oldest to newest
        for i, positions in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))

            for center in positions:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )

        return frame
