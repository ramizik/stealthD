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

    Uses cv2.findHomography (not getPerspectiveTransform) because the number of
    detected keypoints changes as the camera moves. findHomography computes the
    homography matrix that best maps the source points to target points using RANSAC.

    This allows us to handle varying numbers of visible keypoints (4 to 32) as the
    camera pans and zooms during the match.
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

        # Use findHomography with RANSAC for robustness
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, 5.0)

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
    Coordinate system: 12000 x 7000 units (for convenient integer coordinates)
    """

    def __init__(self):
        """Initialize soccer pitch configuration with 32 reference points."""
        # Pitch dimensions in centimeters
        self.width = 7000  # 68m = 6800cm, scaled to 7000 units
        self.length = 12000  # 105m = 10500cm, scaled to 12000 units
        self.penalty_box_width = 4100  # 40.3m scaled
        self.penalty_box_length = 2015  # 16.5m scaled
        self.goal_box_width = 1832  # 18.32m scaled
        self.goal_box_length = 550  # 5.5m scaled
        self.centre_circle_radius = 915  # 9.15m scaled
        self.penalty_spot_distance = 1100  # 11m scaled

        # 32 vertices as defined in research
        self.vertices = [
            (0, 0),  # 1: Top-left corner
            (0, (self.width - self.penalty_box_width) / 2),  # 2: Left penalty area top
            (0, (self.width - self.goal_box_width) / 2),  # 3: Left goal area top
            (0, (self.width + self.goal_box_width) / 2),  # 4: Left goal area bottom
            (0, (self.width + self.penalty_box_width) / 2),  # 5: Left penalty area bottom
            (0, self.width),  # 6: Bottom-left corner
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7: Left goal box top-right
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8: Left goal box bottom-right
            (self.penalty_spot_distance, self.width / 2),  # 9: Left penalty spot
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10: Left penalty box top-right
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13: Left penalty box bottom-right
            (self.length / 2, 0),  # 14: Center line top
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15: Center circle top
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16: Center circle bottom
            (self.length / 2, self.width),  # 17: Center line bottom
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18: Right penalty box top-left
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21: Right penalty box bottom-left
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22: Right penalty spot
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23: Right goal box top-left
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24: Right goal box bottom-left
            (self.length, 0),  # 25: Top-right corner
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26: Right penalty area top
            (self.length, (self.width - self.goal_box_width) / 2),  # 27: Right goal area top
            (self.length, (self.width + self.goal_box_width) / 2),  # 28: Right goal area bottom
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29: Right penalty area bottom
            (self.length, self.width),  # 30: Bottom-right corner
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31: Center circle left
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32: Center circle right
        ]

        # Convert to numpy array
        self.vertices = np.array(self.vertices, dtype=np.float32)

        # Define edges for visualization
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
            (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
            (15, 16), (17, 18), (18, 19), (19, 20), (22, 23),
            (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
            (0, 13), (1, 9), (2, 6), (3, 7), (4, 12), (5, 16),
            (13, 24), (17, 25), (22, 26), (23, 27), (20, 28), (16, 29)
        ]

        # Labels for each vertex
        self.labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31", "32"
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
    radius = int(config.center_circle_radius * scale_x)
    cv2.circle(pitch, center_point, radius, line_color, line_thickness)
    cv2.circle(pitch, center_point, 5, line_color, -1)  # Center spot

    # Draw left penalty area
    penalty_top = (config.width - config.penalty_area_width) / 2
    penalty_bottom = (config.width + config.penalty_area_width) / 2
    cv2.rectangle(pitch,
                  to_image_coords(0, penalty_top),
                  to_image_coords(config.penalty_area_length, penalty_bottom),
                  line_color, line_thickness)

    # Draw left goal area
    goal_top = (config.width - config.goal_area_width) / 2
    goal_bottom = (config.width + config.goal_area_width) / 2
    cv2.rectangle(pitch,
                  to_image_coords(0, goal_top),
                  to_image_coords(config.goal_area_length, goal_bottom),
                  line_color, line_thickness)

    # Draw left penalty arc
    penalty_spot_x = 1320  # 11m penalty spot
    penalty_spot = to_image_coords(penalty_spot_x, center_y)
    cv2.circle(pitch, penalty_spot, 5, line_color, -1)  # Penalty spot
    # Draw arc (approximated with ellipse)
    arc_radius = int(1098 * scale_x)  # 9.15m radius
    cv2.ellipse(pitch, penalty_spot, (arc_radius, arc_radius),
                0, -55, 55, line_color, line_thickness)

    # Draw right penalty area
    cv2.rectangle(pitch,
                  to_image_coords(config.length - config.penalty_area_length, penalty_top),
                  to_image_coords(config.length, penalty_bottom),
                  line_color, line_thickness)

    # Draw right goal area
    cv2.rectangle(pitch,
                  to_image_coords(config.length - config.goal_area_length, goal_top),
                  to_image_coords(config.length, goal_bottom),
                  line_color, line_thickness)

    # Draw right penalty arc
    penalty_spot_x_right = config.length - 1320
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
