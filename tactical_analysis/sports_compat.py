"""
Custom implementation to replace the missing 'sports' module functionality.

This module provides ViewTransformer, SoccerPitchConfiguration, and draw_pitch
functions that were originally part of the sports module in older supervision versions.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class ViewTransformer:
    """
    ViewTransformer for perspective transformation using homography.

    ROBOFLOW/SPORTS APPROACH:
    Uses cv2.findHomography() with RANSAC (default) for robust outlier rejection.
    This is critical for handling noisy keypoint detections in real-world footage.

    The roboflow/sports library uses:
        self.m, _ = cv2.findHomography(source, target)

    Which defaults to cv2.RANSAC with threshold 3.0 when more than 4 points.

    Uses findHomography (not getPerspectiveTransform) to handle varying numbers
    of visible keypoints (4 to 32) as camera pans and zooms during the match.
    """

    def __init__(
            self,
            source: np.ndarray,
            target: np.ndarray
    ) -> None:
        """
        Initialize the ViewTransformer with source and target points.

        Args:
            source: Source points for homography calculation (pixel coordinates)
            target: Target points for homography calculation (pitch coordinates)

        Raises:
            ValueError: If source and target do not have the same shape or if they are
                not 2D coordinates, or if homography cannot be calculated.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        if len(source) < 4:
            raise ValueError(f"Need at least 4 point pairs for homography, got {len(source)}")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # ROBOFLOW/SPORTS: Use cv2.findHomography with default RANSAC
        # When called without method parameter and >4 points, OpenCV uses RANSAC
        # with threshold=3.0 automatically for robustness
        self.m, _ = cv2.findHomography(source, target)

        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(
            self,
            points: np.ndarray
    ) -> np.ndarray:
        """
        Transform the given points using the homography matrix.

        Args:
            points: Points to be transformed (N, 2) array

        Returns:
            Transformed points (N, 2) array

        Raises:
            ValueError: If points are not 2D coordinates.
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: np.ndarray,
            resolution_wh: Tuple[int, int]
    ) -> np.ndarray:
        """
        Transform the given image using the homography matrix.

        Args:
            image: Image to be transformed
            resolution_wh: Width and height of the output image

        Returns:
            Transformed image

        Raises:
            ValueError: If the image is not either grayscale or color.
        """
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        return cv2.warpPerspective(image, self.m, resolution_wh)


class SoccerPitchConfiguration:
    """
    Soccer Pitch Configuration with 32 dense keypoints.

    Based on research: We define 32 characteristic points densely enough so that
    at any time - even when the camera is tightly following the action - at least
    four characteristic points are visible for homography calculation.

    Pitch dimensions: 105m x 68m (FIFA standard)
    Coordinate system: 12000 x 7000 units (centimeters - ROBOFLOW/SPORTS STANDARD)

    CRITICAL: Uses centimeters (12000 x 7000) to match roboflow/sports exactly.
    This is the correct scale - homography works with these coordinates.
    """

    def __init__(self):
        """Initialize soccer pitch configuration with 32 reference points."""
        # Pitch dimensions in centimeters (ROBOFLOW/SPORTS STANDARD)
        self.width = 7000  # [cm]
        self.length = 12000  # [cm]
        self.penalty_box_width = 4100  # [cm]
        self.penalty_box_length = 2015  # [cm]
        self.goal_box_width = 1832  # [cm]
        self.goal_box_length = 550  # [cm]
        self.centre_circle_radius = 915  # [cm]
        self.penalty_spot_distance = 1100  # [cm]        # Define edges for visualization
        self.edges = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
            (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
            (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
        ]

        # Labels for each vertex
        self.labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
            "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
            "14", "19"
        ]

        # Colors for visualization
        self.colors = [
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
            "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
            "#00BFFF", "#00BFFF"
        ]

    @property
    def vertices(self):
        """Return the 32 vertices as a list of tuples."""
        return [
            (0, 0),  # 1
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32
        ]


def draw_pitch(config: SoccerPitchConfiguration,
               width: int = 1050,
               height: int = 680,
               background_color: Tuple[int, int, int] = (34, 139, 34),
               line_color: Tuple[int, int, int] = (255, 255, 255),
               line_thickness: int = 2) -> np.ndarray:
    """
    Draw a soccer pitch visualization.

    Args:
        config: SoccerPitchConfiguration object with pitch dimensions
        width: Width of the output image
        height: Height of the output image
        background_color: RGB color for pitch background (default: green)
        line_color: RGB color for pitch lines (default: white)
        line_thickness: Thickness of pitch lines

    Returns:
        Numpy array representing the pitch image (height, width, 3)
    """
    # Create pitch background
    pitch = np.ones((height, width, 3), dtype=np.uint8)
    pitch[:] = background_color

    # Scale factors for converting pitch coordinates to image coordinates
    scale_x = width / config.length
    scale_y = height / config.width

    def to_image_coords(x, y):
        """Convert pitch coordinates to image coordinates."""
        return (int(x * scale_x), int(y * scale_y))

    # Draw outer boundary
    cv2.rectangle(pitch,
                  to_image_coords(0, 0),
                  to_image_coords(config.length, config.width),
                  line_color, line_thickness)

    # Draw center line
    center_x = config.length / 2
    cv2.line(pitch,
             to_image_coords(center_x, 0),
             to_image_coords(center_x, config.width),
             line_color, line_thickness)

    # Draw center circle
    center_y = config.width / 2
    center_point = to_image_coords(center_x, center_y)
    radius = int(config.centre_circle_radius * scale_x)
    cv2.circle(pitch, center_point, radius, line_color, line_thickness)
    cv2.circle(pitch, center_point, 5, line_color, -1)  # Center spot

    # Draw left penalty area
    penalty_top = (config.width - config.penalty_box_width) / 2
    penalty_bottom = (config.width + config.penalty_box_width) / 2
    cv2.rectangle(pitch,
                  to_image_coords(0, penalty_top),
                  to_image_coords(config.penalty_box_length, penalty_bottom),
                  line_color, line_thickness)

    # Draw left goal area
    goal_top = (config.width - config.goal_box_width) / 2
    goal_bottom = (config.width + config.goal_box_width) / 2
    cv2.rectangle(pitch,
                  to_image_coords(0, goal_top),
                  to_image_coords(config.goal_box_length, goal_bottom),
                  line_color, line_thickness)

    # Draw left penalty arc
    penalty_spot_x = config.penalty_spot_distance
    penalty_spot = to_image_coords(penalty_spot_x, center_y)
    cv2.circle(pitch, penalty_spot, 5, line_color, -1)  # Penalty spot
    # Draw arc (approximated with ellipse)
    arc_radius = int(config.centre_circle_radius * scale_x)  # 9.15m radius
    cv2.ellipse(pitch, penalty_spot, (arc_radius, arc_radius),
                0, -55, 55, line_color, line_thickness)

    # Draw right penalty area
    cv2.rectangle(pitch,
                  to_image_coords(config.length - config.penalty_box_length, penalty_top),
                  to_image_coords(config.length, penalty_bottom),
                  line_color, line_thickness)

    # Draw right goal area
    cv2.rectangle(pitch,
                  to_image_coords(config.length - config.goal_box_length, goal_top),
                  to_image_coords(config.length, goal_bottom),
                  line_color, line_thickness)

    # Draw right penalty arc
    penalty_spot_x_right = config.length - config.penalty_spot_distance
    penalty_spot_right = to_image_coords(penalty_spot_x_right, center_y)
    cv2.circle(pitch, penalty_spot_right, 5, line_color, -1)  # Penalty spot
    cv2.ellipse(pitch, penalty_spot_right, (arc_radius, arc_radius),
                0, 125, 235, line_color, line_thickness)

    # Draw corner arcs
    corner_radius = int(120 * scale_x)  # 1m corner arc radius
    # Top-left
    cv2.ellipse(pitch, to_image_coords(0, 0), (corner_radius, corner_radius),
                0, 0, 90, line_color, line_thickness)
    # Top-right
    cv2.ellipse(pitch, to_image_coords(config.length, 0), (corner_radius, corner_radius),
                0, 90, 180, line_color, line_thickness)
    # Bottom-left
    cv2.ellipse(pitch, to_image_coords(0, config.width), (corner_radius, corner_radius),
                0, 270, 360, line_color, line_thickness)
    # Bottom-right
    cv2.ellipse(pitch, to_image_coords(config.length, config.width), (corner_radius, corner_radius),
                0, 180, 270, line_color, line_thickness)

    return pitch
