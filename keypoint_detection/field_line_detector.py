"""Field Line Detection for Soccer Field Analysis.

Detects field lines, intersections, and geometric features using computer vision
to enable robust homography mapping for tactical camera recordings.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class FieldLineDetector:
    """
    Detect soccer field lines and extract geometric keypoints.

    Optimized for tactical camera recordings with panning motion.
    Uses Hough line detection and geometric analysis to find field features.
    """

    def __init__(self, min_line_length: int = 50, max_line_gap: int = 10):
        """
        Initialize field line detector.

        Args:
            min_line_length: Minimum line length for Hough detection
            max_line_gap: Maximum gap between line segments
        """
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

        # Hough parameters
        self.hough_threshold = 80
        self.hough_rho = 1
        self.hough_theta = np.pi / 180

        # Line filtering
        self.min_angle_diff = 10  # degrees, to separate horizontal/vertical
        self.intersection_tolerance = 20  # pixels

    def detect_field_features(self, frame: np.ndarray) -> Dict:
        """
        Detect field lines and extract keypoints for homography.

        Args:
            frame: Input video frame (BGR)

        Returns:
            Dictionary containing:
                - 'keypoints': List of (x, y) intersection points
                - 'lines': Detected line segments
                - 'horizontal_lines': Filtered horizontal lines
                - 'vertical_lines': Filtered vertical lines
                - 'confidence': Quality score [0, 1]
                - 'field_mask': Binary mask of detected field
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract white lines (field markings)
        white_mask = self._extract_white_lines(frame, gray)

        # Hough line detection
        lines = cv2.HoughLinesP(
            white_mask,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None or len(lines) == 0:
            print(f"[Field Line Detection] WARNING: No lines detected in frame")
            return self._empty_result()

        # Separate horizontal and vertical lines
        horizontal_lines, vertical_lines = self._separate_lines(lines)

        # Find line intersections (keypoints)
        keypoints = self._extract_intersections(horizontal_lines, vertical_lines)

        # Calculate confidence based on number of features
        confidence = self._calculate_confidence(keypoints, lines)

        print(f"[Field Line Detection] Detected {len(lines)} lines, "
              f"{len(keypoints)} keypoints, confidence: {confidence:.3f}")

        return {
            'keypoints': keypoints,
            'lines': lines,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'confidence': confidence,
            'field_mask': white_mask
        }

    def _extract_white_lines(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Extract white field lines from frame.

        Uses both HSV color filtering and intensity thresholding.
        """
        # Method 1: HSV-based white detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White line mask (high saturation filtering)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        white_mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # Method 2: Intensity-based thresholding
        _, white_mask_intensity = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Combine both methods
        white_mask = cv2.bitwise_or(white_mask_hsv, white_mask_intensity)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Edge detection for better line extraction
        edges = cv2.Canny(white_mask, 50, 150)

        return edges

    def _separate_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """
        Separate lines into horizontal and vertical groups.

        Args:
            lines: Array of line segments [[x1, y1, x2, y2], ...]

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan(dy / dx)))

            # Classify as horizontal or vertical
            if angle < (90 - self.min_angle_diff):
                horizontal_lines.append(line[0])
            elif angle > (90 - self.min_angle_diff):
                vertical_lines.append(line[0])

        return horizontal_lines, vertical_lines

    def _extract_intersections(self, horizontal_lines: List, vertical_lines: List) -> List[Tuple[float, float]]:
        """
        Find intersection points between horizontal and vertical lines.

        Args:
            horizontal_lines: List of horizontal line segments
            vertical_lines: List of vertical line segments

        Returns:
            List of (x, y) intersection points
        """
        intersections = []

        for h_line in horizontal_lines:
            x1h, y1h, x2h, y2h = h_line

            for v_line in vertical_lines:
                x1v, y1v, x2v, y2v = v_line

                # Find intersection point
                intersection = self._line_intersection(
                    (x1h, y1h), (x2h, y2h),
                    (x1v, y1v), (x2v, y2v)
                )

                if intersection is not None:
                    intersections.append(intersection)

        # Remove duplicate intersections (within tolerance)
        intersections = self._remove_duplicate_points(intersections)

        return intersections

    def _line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Find intersection point of two line segments.

        Args:
            p1, p2: Points defining first line
            p3, p4: Points defining second line

        Returns:
            Intersection point (x, y) or None if lines don't intersect
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Check if intersection is within line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (float(x), float(y))

        return None

    def _remove_duplicate_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove duplicate points within tolerance."""
        if not points:
            return []

        unique_points = []

        for point in points:
            is_duplicate = False
            for existing in unique_points:
                dist = np.sqrt((point[0] - existing[0])**2 + (point[1] - existing[1])**2)
                if dist < self.intersection_tolerance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(point)

        return unique_points

    def _calculate_confidence(self, keypoints: List, lines: np.ndarray) -> float:
        """
        Calculate confidence score based on detected features.

        Args:
            keypoints: List of detected intersection points
            lines: Array of detected lines

        Returns:
            Confidence score [0, 1]
        """
        if lines is None or len(lines) == 0:
            return 0.0

        # Base confidence on number of keypoints and lines
        keypoint_score = min(len(keypoints) / 20.0, 1.0)  # 20+ keypoints = max score
        line_score = min(len(lines) / 50.0, 1.0)  # 50+ lines = max score

        # Weighted combination
        confidence = 0.7 * keypoint_score + 0.3 * line_score

        return float(confidence)

    def _empty_result(self) -> Dict:
        """Return empty result when no lines detected."""
        return {
            'keypoints': [],
            'lines': None,
            'horizontal_lines': [],
            'vertical_lines': [],
            'confidence': 0.0,
            'field_mask': None
        }
