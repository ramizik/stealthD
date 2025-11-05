"""
Confidence Visualizer for Global Homography Mapper.

Provides visualization tools to show field coverage and confidence maps
for debugging and validating the global homography quality.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
import cv2
from typing import Tuple, Optional


class ConfidenceVisualizer:
    """
    Visualize confidence maps and field coverage for global homography mapper.

    Creates heat maps showing which field regions have good keypoint coverage
    and identifies low-confidence zones.
    """

    def __init__(self, field_dimensions: Tuple[float, float] = (105, 68)):
        """
        Initialize the confidence visualizer.

        Args:
            field_dimensions: (width, height) in meters (default: 105x68)
        """
        self.field_width = field_dimensions[0]
        self.field_height = field_dimensions[1]

    def visualize_confidence_map(self, global_mapper, output_path: Optional[str] = None,
                                 width: int = 1050, height: int = 680) -> np.ndarray:
        """
        Create a heat map visualization of the confidence map.

        Args:
            global_mapper: GlobalHomographyMapper object
            output_path: Optional path to save the visualization (e.g., 'confidence_map.png')
            width: Width of output image in pixels
            height: Height of output image in pixels

        Returns:
            Numpy array containing the visualization image (height, width, 3)
        """
        # Get confidence grid from mapper
        confidence_grid = global_mapper.get_field_confidence_map()

        if confidence_grid is None or confidence_grid.size == 0:
            print("[Confidence Visualizer] No confidence map available")
            return self._create_empty_visualization(width, height)

        # Resize confidence grid to match output dimensions
        confidence_resized = cv2.resize(confidence_grid, (width, height),
                                       interpolation=cv2.INTER_LINEAR)

        # Convert to color heat map (BGR)
        # Low confidence = blue/cold, High confidence = red/hot
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                conf = confidence_resized[y, x]

                # Color mapping:
                # 0.0-0.3: Blue (low confidence)
                # 0.3-0.6: Green (medium confidence)
                # 0.6-1.0: Red (high confidence)
                if conf < 0.3:
                    # Blue to Cyan
                    heatmap[y, x] = [int(255 * (1 - conf/0.3)), 0, 255]
                elif conf < 0.6:
                    # Cyan to Green
                    t = (conf - 0.3) / 0.3
                    heatmap[y, x] = [0, int(255 * t), int(255 * (1 - t))]
                else:
                    # Green to Red
                    t = (conf - 0.6) / 0.4
                    heatmap[y, x] = [0, int(255 * (1 - t)), int(255 * t)]

        # Add field lines overlay
        self._draw_field_lines(heatmap, width, height)

        # Add legend
        self._draw_legend(heatmap)

        # Add statistics overlay
        self._draw_statistics(heatmap, global_mapper, confidence_grid)

        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, heatmap)
            print(f"[Confidence Visualizer] Saved confidence map to: {output_path}")

        return heatmap

    def _create_empty_visualization(self, width: int, height: int) -> np.ndarray:
        """Create an empty/error visualization."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, "No confidence data available", (width//4, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    def _draw_field_lines(self, img: np.ndarray, width: int, height: int):
        """Draw field lines overlay on the heat map."""
        # Scale factors
        scale_x = width / self.field_width
        scale_y = height / self.field_height

        color = (255, 255, 255)  # White
        thickness = 1

        # Outer boundary
        cv2.rectangle(img, (0, 0), (width-1, height-1), color, thickness)

        # Center line
        center_x = int(self.field_width / 2 * scale_x)
        cv2.line(img, (center_x, 0), (center_x, height-1), color, thickness)

        # Center circle (approximate)
        center_y = int(self.field_height / 2 * scale_y)
        radius = int(9.15 * scale_x)  # 9.15m radius
        cv2.circle(img, (center_x, center_y), radius, color, thickness)

        # Penalty areas (approximate)
        penalty_width = int(16.5 * scale_x)
        penalty_height = int(40.3 * scale_y)
        penalty_y_offset = int((self.field_height - 40.3) / 2 * scale_y)

        # Left penalty area
        cv2.rectangle(img, (0, penalty_y_offset),
                     (penalty_width, penalty_y_offset + penalty_height),
                     color, thickness)

        # Right penalty area
        cv2.rectangle(img, (width - penalty_width, penalty_y_offset),
                     (width-1, penalty_y_offset + penalty_height),
                     color, thickness)

    def _draw_legend(self, img: np.ndarray):
        """Draw color legend on the image."""
        legend_height = 30
        legend_width = 200
        legend_x = 20
        legend_y = img.shape[0] - 60

        # Draw legend background
        cv2.rectangle(img, (legend_x-5, legend_y-5),
                     (legend_x + legend_width + 5, legend_y + legend_height + 5),
                     (50, 50, 50), -1)

        # Draw gradient
        for i in range(legend_width):
            conf = i / legend_width
            if conf < 0.3:
                color = (int(255 * (1 - conf/0.3)), 0, 255)
            elif conf < 0.6:
                t = (conf - 0.3) / 0.3
                color = (0, int(255 * t), int(255 * (1 - t)))
            else:
                t = (conf - 0.6) / 0.4
                color = (0, int(255 * (1 - t)), int(255 * t))

            cv2.line(img, (legend_x + i, legend_y),
                    (legend_x + i, legend_y + legend_height),
                    color, 1)

        # Add text labels
        cv2.putText(img, "Low", (legend_x, legend_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(img, "High", (legend_x + legend_width - 30, legend_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_statistics(self, img: np.ndarray, global_mapper, confidence_grid: np.ndarray):
        """Draw statistics overlay on the image."""
        stats = global_mapper.get_statistics()

        # Prepare text
        text_lines = [
            f"Keypoints: {stats.get('total_keypoints_accumulated', 0)}",
            f"Unique: {stats.get('unique_keypoints_detected', 0)}",
            f"High Coverage: {stats.get('high_confidence_coverage_percent', 0):.1f}%"
        ]

        # Draw background box
        box_x = 20
        box_y = 20
        box_width = 250
        box_height = 80
        cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height),
                     (50, 50, 50), -1)

        # Draw text
        y_offset = box_y + 25
        for line in text_lines:
            cv2.putText(img, line, (box_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

    def print_coverage_report(self, global_mapper):
        """
        Print a detailed text report of field coverage.

        Args:
            global_mapper: GlobalHomographyMapper object
        """
        confidence_grid = global_mapper.get_field_confidence_map()

        if confidence_grid is None or confidence_grid.size == 0:
            print("\n[Coverage Report] No confidence data available")
            return

        stats = global_mapper.get_statistics()
        grid_dims = stats.get('grid_dimensions', (0, 0))

        print("\n" + "="*70)
        print("FIELD COVERAGE REPORT")
        print("="*70)

        print(f"\nGlobal Homography Statistics:")
        print(f"  Total keypoints accumulated: {stats.get('total_keypoints_accumulated', 0)}")
        print(f"  Unique keypoints detected: {stats.get('unique_keypoints_detected', 0)}")
        print(f"  Frames with keypoints: {stats.get('frames_with_keypoints', 0)}/{stats.get('total_frames_processed', 0)}")

        # Analyze confidence grid
        high_conf = np.sum(confidence_grid > 0.7)
        medium_conf = np.sum((confidence_grid > 0.4) & (confidence_grid <= 0.7))
        low_conf = np.sum(confidence_grid <= 0.4)
        total_cells = confidence_grid.size

        print(f"\nConfidence Map ({grid_dims[0]}x{grid_dims[1]} cells, 5m resolution):")
        print(f"  High confidence (>0.7):   {high_conf:4d} cells ({100*high_conf/total_cells:5.1f}%)")
        print(f"  Medium confidence (0.4-0.7): {medium_conf:4d} cells ({100*medium_conf/total_cells:5.1f}%)")
        print(f"  Low confidence (<0.4):    {low_conf:4d} cells ({100*low_conf/total_cells:5.1f}%)")

        # Identify problematic regions
        print(f"\nLow-Confidence Regions:")
        grid_h, grid_w = confidence_grid.shape
        cell_size = 5.0  # meters

        low_conf_regions = []
        for y in range(grid_h):
            for x in range(grid_w):
                if confidence_grid[y, x] < 0.4:
                    x_meters = x * cell_size
                    y_meters = y * cell_size
                    low_conf_regions.append((x_meters, y_meters, confidence_grid[y, x]))

        if low_conf_regions:
            # Group by field thirds
            left_third = [r for r in low_conf_regions if r[0] < 35]
            middle_third = [r for r in low_conf_regions if 35 <= r[0] < 70]
            right_third = [r for r in low_conf_regions if r[0] >= 70]

            print(f"  Left third (0-35m):   {len(left_third)} low-confidence cells")
            print(f"  Middle third (35-70m): {len(middle_third)} low-confidence cells")
            print(f"  Right third (70-105m): {len(right_third)} low-confidence cells")
        else:
            print(f"  None - full field coverage!")

        print("="*70 + "\n")
