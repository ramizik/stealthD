"""
Track utility functions for converting and enhancing track data structures.
"""

import numpy as np
from typing import Dict
from utils import get_center_of_bbox


def convert_tracks_to_enhanced_format(tracks: Dict) -> Dict:
    """
    Convert tracks from simple bbox format to enhanced format with position_adjusted and bbox.

    Input format:
    {
        'object_type': {
            frame_idx: {
                track_id: [x1, y1, x2, y2]  # or just bbox list
            }
        }
    }

    Output format:
    {
        'object_type': {
            frame_idx: {
                track_id: {
                    'bbox': [x1, y1, x2, y2],
                    'position_adjusted': [center_x, center_y]
                }
            }
        }
    }

    Args:
        tracks: Dictionary of tracks in simple bbox format

    Returns:
        Dictionary of tracks in enhanced format
    """
    enhanced_tracks = {}

    # Metadata keys to skip (not track data)
    metadata_keys = ['player_classids', 'player_is_goalkeeper']

    for object_type, object_tracks in tracks.items():
        # Skip metadata keys
        if object_type in metadata_keys:
            continue

        # Skip if object_tracks is not a dictionary
        if not isinstance(object_tracks, dict):
            continue

        enhanced_tracks[object_type] = {}

        for frame_idx, frame_data in object_tracks.items():
            # Skip invalid frame indices
            if not isinstance(frame_idx, int):
                continue

            enhanced_tracks[object_type][frame_idx] = {}

            # Handle ball tracks (different structure: {frame_idx: [x1, y1, x2, y2]})
            if object_type == 'ball':
                if isinstance(frame_data, list) and len(frame_data) >= 4:
                    bbox = frame_data
                    if bbox is not None and not any(b is None for b in bbox):
                        center_x, center_y = get_center_of_bbox(bbox)
                        # Ball doesn't have track_id, use 0 as default
                        enhanced_tracks[object_type][frame_idx][0] = {
                            'bbox': bbox,
                            'position_adjusted': [center_x, center_y]
                        }
                continue

            # Handle player and referee tracks (structure: {frame_idx: {track_id: [x1, y1, x2, y2]}})
            if not isinstance(frame_data, dict):
                continue

            for track_id, track_data in frame_data.items():
                # Skip invalid entries
                if track_id == -1:
                    continue

                # If track_data is already a dict with bbox, use it
                if isinstance(track_data, dict):
                    if 'bbox' in track_data:
                        bbox = track_data['bbox']
                    elif 'position_adjusted' in track_data:
                        # Try to reconstruct bbox from position (not ideal, but handle it)
                        continue
                    else:
                        continue
                # If track_data is a list (bbox), convert it
                elif isinstance(track_data, list) and len(track_data) >= 4:
                    bbox = track_data
                else:
                    continue

                # Skip None bboxes
                if bbox is None or any(b is None for b in bbox):
                    continue

                # Calculate center position
                center_x, center_y = get_center_of_bbox(bbox)

                # Create enhanced track info
                enhanced_tracks[object_type][frame_idx][track_id] = {
                    'bbox': bbox,
                    'position_adjusted': [center_x, center_y]
                }

    return enhanced_tracks
