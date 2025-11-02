"""
Custom implementation to replace the missing 'sports' module functionality.

This module provides ViewTransformer, SoccerPitchConfiguration, and draw_pitch
functions that were originally part of the sports module in older supervision versions.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class ViewTransformer:
    """
    Custom ViewTransformer to replace sports.common.view.ViewTransformer.
    
    Uses OpenCV's homography transformation to convert between coordinate systems
    (e.g., camera view to pitch view).
    """
    
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize the ViewTransformer with source and target points.
        
        Args:
            source: Source coordinate points as numpy array (N, 2)
            target: Target coordinate points as numpy array (N, 2)
        """
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        
        # Compute homography matrix
        if len(source) >= 4 and len(target) >= 4:
            self.matrix, self.mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        else:
            raise ValueError(f"Need at least 4 point correspondences, got {len(source)}")
            
        if self.matrix is None:
            raise ValueError("Could not compute homography matrix from given points")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points using the computed homography matrix.
        
        Args:
            points: Points to transform as numpy array (N, 2)
            
        Returns:
            Transformed points as numpy array (N, 2)
        """
        if len(points) == 0:
            return np.array([]).reshape(0, 2)
        
        points = np.array(points, dtype=np.float32)
        
        # Reshape for cv2.perspectiveTransform (needs shape (N, 1, 2))
        points_reshaped = points.reshape(-1, 1, 2)
        
        # Apply perspective transformation
        transformed = cv2.perspectiveTransform(points_reshaped, self.matrix)
        
        # Reshape back to (N, 2)
        return transformed.reshape(-1, 2)


class SoccerPitchConfiguration:
    """
    Custom SoccerPitchConfiguration to replace sports.configs.soccer.SoccerPitchConfiguration.
    
    Defines a FIFA-standard soccer pitch with all reference points including:
    - Corner points
    - Penalty areas
    - Goal areas
    - Center circle
    - Touchlines and goal lines
    """
    
    def __init__(self):
        """Initialize soccer pitch configuration with FIFA standard dimensions."""
        # Pitch dimensions in units (scaled representation)
        # Standard FIFA pitch: 105m x 68m
        # We use 12000 x 7000 units for convenient integer coordinates
        self.length = 12000  # Full pitch length
        self.width = 7000    # Full pitch width
        
        # Penalty area dimensions
        self.penalty_area_length = 1980  # 16.5m scaled
        self.penalty_area_width = 4860   # 40.3m scaled
        
        # Goal area dimensions
        self.goal_area_length = 660      # 5.5m scaled
        self.goal_area_width = 2200      # 18.3m scaled
        
        # Center circle radius
        self.center_circle_radius = 1098  # 9.15m scaled
        
        # Define all vertices (35 reference points)
        self.vertices = self._create_pitch_vertices()
    
    def _create_pitch_vertices(self) -> np.ndarray:
        """
        Create all pitch reference points.
        
        Returns:
            Array of shape (35, 2) with all pitch vertices
        """
        vertices = []
        
        # Corner points (0-3)
        vertices.append([0, 0])                          # 0: Top-left corner
        vertices.append([0, self.width])                 # 1: Bottom-left corner (was index 5)
        vertices.append([self.length, 0])                # 2: Top-right corner (was index 24)
        vertices.append([self.length, self.width])       # 3: Bottom-right corner (was index 29)
        
        # Left penalty area (4-7)
        penalty_top = (self.width - self.penalty_area_width) / 2
        penalty_bottom = (self.width + self.penalty_area_width) / 2
        vertices.append([self.penalty_area_length, penalty_top])        # 4: Left penalty top-right
        vertices.append([0, penalty_top])                               # 5: Left penalty top-left (was 1)
        vertices.append([0, penalty_bottom])                            # 6: Left penalty bottom-left (was 12)
        vertices.append([self.penalty_area_length, penalty_bottom])     # 7: Left penalty bottom-right
        
        # Left goal area (8-11)
        goal_top = (self.width - self.goal_area_width) / 2
        goal_bottom = (self.width + self.goal_area_width) / 2
        vertices.append([self.goal_area_length, goal_top])              # 8: Left goal top-right
        vertices.append([0, goal_top])                                  # 9: Left goal top-left (was 2)
        vertices.append([0, goal_bottom])                               # 10: Left goal bottom-left (was 3)
        vertices.append([self.goal_area_length, goal_bottom])           # 11: Left goal bottom-right
        
        # Center line and circle (12-15)
        center_x = self.length / 2
        center_y = self.width / 2
        vertices.append([center_x, 0])                                  # 12: Center line top (was 13)
        vertices.append([center_x, center_y - self.center_circle_radius])  # 13: Center circle top (was 14)
        vertices.append([center_x, center_y + self.center_circle_radius])  # 14: Center circle bottom (was 15)
        vertices.append([center_x, self.width])                         # 15: Center line bottom (was 16)
        
        # Right goal area (16-19) - mirrored from left
        vertices.append([self.length - self.goal_area_length, goal_top])   # 16: Right goal top-left
        vertices.append([self.length, goal_top])                            # 17: Right goal top-right (was 17)
        vertices.append([self.length, goal_bottom])                         # 18: Right goal bottom-right (was 20)
        vertices.append([self.length - self.goal_area_length, goal_bottom]) # 19: Right goal bottom-left
        
        # Right penalty area (20-23) - mirrored from left
        vertices.append([self.length - self.penalty_area_length, penalty_top])     # 20: Right penalty top-left
        vertices.append([self.length, penalty_top])                                # 21: Right penalty top-right (was 25)
        vertices.append([self.length, penalty_bottom])                             # 22: Right penalty bottom-right (was 28)
        vertices.append([self.length - self.penalty_area_length, penalty_bottom])  # 23: Right penalty bottom-left
        
        # Additional touchline points (24-29)
        vertices.append([0, self.width])                 # 24: Bottom-left corner duplicate
        vertices.append([self.length, 0])                # 25: Top-right corner duplicate
        vertices.append([0, goal_top])                   # 26: Left goal top-left duplicate
        vertices.append([0, goal_bottom])                # 27: Left goal bottom-left duplicate
        vertices.append([self.length, penalty_bottom])   # 28: Right penalty bottom-right duplicate
        vertices.append([self.length, self.width])       # 29: Bottom-right corner duplicate
        
        # Circle points (30-31)
        vertices.append([center_x - self.center_circle_radius, center_y])  # 30: Center circle left
        vertices.append([center_x + self.center_circle_radius, center_y])  # 31: Center circle right
        
        # Penalty arc points (32-34)
        penalty_arc_x_left = self.penalty_area_length
        penalty_arc_x_right = self.length - self.penalty_area_length
        vertices.append([penalty_arc_x_left, center_y])         # 32: Left penalty arc rightmost
        vertices.append([center_x, center_y])                   # 33: Center point
        vertices.append([penalty_arc_x_right, center_y])        # 34: Right penalty arc leftmost
        
        return np.array(vertices, dtype=np.float32)


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

