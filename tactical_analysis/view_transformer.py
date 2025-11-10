"""
Simple ViewTransformer using fixed vertices and perspective transformation.

This is a simpler alternative to the homography-based ViewTransformer that uses
fixed pixel vertices and getPerspectiveTransform for court coordinate transformation.
"""

import numpy as np
import cv2


class ViewTransformer:
    """
    Simple ViewTransformer using fixed vertices and getPerspectiveTransform.

    This transformer uses 4 fixed pixel vertices to create a perspective
    transformation matrix for converting pixel coordinates to court coordinates.
    """

    def __init__(self, pixel_vertices=None, target_vertices=None, court_width=68, court_length=23.32):
        """
        Initialize the ViewTransformer.

        Args:
            pixel_vertices: Array of 4 pixel coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                          If None, uses default vertices
            target_vertices: Array of 4 target court coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                           If None, uses default court coordinates
            court_width: Width of the court in meters (default: 68m for soccer)
            court_length: Length of the court in meters (default: 23.32m for futsal)
        """
        self.court_width = court_width
        self.court_length = court_length

        # Default pixel vertices (can be overridden)
        if pixel_vertices is None:
            self.pixel_vertices = np.array([[110, 1035],
                                           [265, 275],
                                           [910, 260],
                                           [1640, 915]])
        else:
            self.pixel_vertices = np.array(pixel_vertices)

        # Default target vertices (court coordinates)
        if target_vertices is None:
            self.target_vertices = np.array([
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width]
            ])
        else:
            self.target_vertices = np.array(target_vertices)

        # Ensure float32 type for OpenCV
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Create perspective transformation matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """
        Transform a single point from pixel coordinates to court coordinates.

        Args:
            point: Point as [x, y] or np.array([x, y])

        Returns:
            Transformed point as np.array([x, y]) or None if transformation fails or result is invalid
        """
        point = np.array(point, dtype=np.float32)
        p = (int(point[0]), int(point[1]))

        # Try to transform even if outside polygon (more lenient)
        # We'll validate the result instead
        try:
            # Transform the point
            reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
            transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
            result = transformed_point.reshape(-1, 2)[0]

            # Validate the result is within reasonable court bounds
            # Allow some margin for edge cases
            margin = 10.0  # 10 meters margin
            if (result[0] >= -margin and result[0] <= self.court_length + margin and
                result[1] >= -margin and result[1] <= self.court_width + margin):
                return result
            else:
                # Result is outside reasonable bounds, return None
                return None
        except Exception:
            return None

    def add_transformed_position_to_tracks(self, tracks):
        """
        Add transformed positions to tracks dictionary.

        This method adds 'position_transformed' to each track entry in the tracks dictionary.
        The tracks structure should be:
        {
            'object_type': {
                frame_num: {
                    track_id: {
                        'position_adjusted': [x, y],
                        ...
                    }
                }
            }
        }

        Args:
            tracks: Dictionary of tracks with structure described above
        """
        for object_type, object_tracks in tracks.items():
            # Skip metadata keys
            if object_type in ['player_classids', 'player_is_goalkeeper']:
                continue

            # Ensure object_tracks is a dictionary
            if not isinstance(object_tracks, dict):
                continue

            for frame_num, frame_data in object_tracks.items():
                # Skip invalid frame indices
                if not isinstance(frame_num, int):
                    continue

                # Handle ball tracks (structure: {frame_idx: {0: dict}} after enhancement)
                if object_type == 'ball':
                    if isinstance(frame_data, dict):
                        # Enhanced format: {0: {'bbox': ..., 'position_adjusted': ...}}
                        for track_id, track_info in frame_data.items():
                            if isinstance(track_info, dict) and 'position_adjusted' in track_info:
                                position = track_info['position_adjusted']
                                position = np.array(position)
                                position_transformed = self.transform_point(position)
                                if position_transformed is not None:
                                    position_transformed = position_transformed.squeeze().tolist()
                                else:
                                    position_transformed = None
                                tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
                    continue

                # Handle player and referee tracks
                if not isinstance(frame_data, dict):
                    continue

                for track_id, track_info in frame_data.items():
                    if not isinstance(track_info, dict):
                        continue
                    if 'position_adjusted' not in track_info:
                        continue

                    position = track_info['position_adjusted']
                    position = np.array(position)

                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    else:
                        position_transformed = None

                    tracks[object_type][frame_num][track_id]['position_transformed'] = position_transformed
