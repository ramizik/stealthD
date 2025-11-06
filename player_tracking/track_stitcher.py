"""
Track Stitcher module for merging fragmented player tracks.

Handles re-identification when ByteTrack loses and re-acquires players.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, List, Set, Tuple

import numpy as np


class TrackStitcher:
    """
    Merges fragmented ByteTrack tracks using appearance and spatial features.

    When ByteTrack loses a player (occlusion, leaves frame), it may assign
    a new ID on re-detection. This class identifies and merges such fragments.
    """

    def __init__(self, max_gap_frames: int = 150, max_position_distance: float = 200):
        """
        Initialize track stitcher.

        Args:
            max_gap_frames: Maximum frames between tracks to consider merging (default: 150 = 5s at 30fps)
            max_position_distance: Maximum pixel distance for track continuation (default: 200px)
        """
        self.max_gap_frames = max_gap_frames
        self.max_position_distance = max_position_distance

    def _get_track_info(self, bytetrack_id: int, player_tracks: Dict,
                        team_assignments: Dict, goalkeeper_metadata: Dict = None) -> Dict:
        """
        Extract information about a specific ByteTrack ID.

        Args:
            bytetrack_id: ByteTrack ID to analyze
            player_tracks: Dictionary {frame_idx: {bytetrack_id: bbox}}
            team_assignments: Dictionary {frame_idx: {bytetrack_id: team_id}}
            goalkeeper_metadata: Optional dictionary {frame_idx: {bytetrack_id: is_goalkeeper}}

        Returns:
            Dictionary with track statistics
        """
        frames = []
        positions = []
        teams = []
        goalkeeper_frames = 0  # Count frames where this ID is marked as goalkeeper

        for frame_idx in sorted(player_tracks.keys()):
            if bytetrack_id in player_tracks[frame_idx] and -1 not in player_tracks[frame_idx]:
                bbox = player_tracks[frame_idx][bytetrack_id]
                frames.append(frame_idx)
                # Get center position
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append((center_x, center_y))

                # Get team
                if frame_idx in team_assignments and bytetrack_id in team_assignments[frame_idx]:
                    teams.append(team_assignments[frame_idx][bytetrack_id])

                # Check goalkeeper status
                if goalkeeper_metadata and frame_idx in goalkeeper_metadata:
                    if bytetrack_id in goalkeeper_metadata[frame_idx]:
                        if goalkeeper_metadata[frame_idx][bytetrack_id]:
                            goalkeeper_frames += 1

        # Determine if this is a goalkeeper track
        # Strategy: Goalkeeper tracks are either:
        # 1. Short fragments (<50 frames) with ANY goalkeeper frames (likely tracking breaks)
        # 2. Longer tracks (>=50 frames) with >50% goalkeeper frames
        is_goalkeeper = False
        if frames and goalkeeper_frames > 0:
            if len(frames) < 50:
                # Short track with any GK frames - likely a goalkeeper fragment
                is_goalkeeper = True
            elif goalkeeper_frames > len(frames) * 0.5:
                # Long track that's mostly goalkeeper
                is_goalkeeper = True

        return {
            'frames': frames,
            'first_frame': frames[0] if frames else None,
            'last_frame': frames[-1] if frames else None,
            'positions': positions,
            'last_position': positions[-1] if positions else None,
            'first_position': positions[0] if positions else None,
            'team': max(set(teams), key=teams.count) if teams else None,  # Most common team
            'total_frames': len(frames),
            'is_goalkeeper': is_goalkeeper,
            'goalkeeper_frame_count': goalkeeper_frames
        }

    def _calculate_position_distance(self, pos1: Tuple[float, float],
                                     pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def find_mergeable_tracks(self, player_tracks: Dict,
                             team_assignments: Dict,
                             goalkeeper_metadata: Dict = None) -> Dict[int, int]:
        """
        Find tracks that should be merged (represent same player).

        Uses spatial continuity and team consistency to identify track fragments.
        IMPORTANT: Goalkeepers are only merged with other goalkeepers, never with field players.

        Args:
            player_tracks: Dictionary {frame_idx: {bytetrack_id: bbox}}
            team_assignments: Dictionary {frame_idx: {bytetrack_id: team_id}}
            goalkeeper_metadata: Optional dictionary {frame_idx: {bytetrack_id: is_goalkeeper}}

        Returns:
            Dictionary {old_bytetrack_id: new_bytetrack_id} for merging
        """
        # Get all unique ByteTrack IDs
        all_ids = set()
        for frame_tracks in player_tracks.values():
            if -1 not in frame_tracks:
                all_ids.update(frame_tracks.keys())

        # Get info for each track
        track_infos = {}
        for track_id in all_ids:
            track_infos[track_id] = self._get_track_info(track_id, player_tracks, team_assignments, goalkeeper_metadata)

        # Debug: Log goalkeeper metadata status
        if goalkeeper_metadata:
            total_gk_frames = sum(1 for frame_data in goalkeeper_metadata.values()
                                 for is_gk in frame_data.values() if is_gk)
            gk_ids_in_metadata = set()
            for frame_data in goalkeeper_metadata.values():
                for tid, is_gk in frame_data.items():
                    if is_gk:
                        gk_ids_in_metadata.add(tid)
            print(f"  Track stitcher received goalkeeper metadata: {total_gk_frames} goalkeeper entries")
            print(f"  ByteTrack IDs marked as goalkeepers in metadata: {sorted(gk_ids_in_metadata)}")
        else:
            print(f"  WARNING: No goalkeeper metadata provided to track stitcher")

        # Log goalkeeper track detection
        goalkeeper_tracks = {tid: info for tid, info in track_infos.items() if info['is_goalkeeper']}
        if goalkeeper_tracks:
            print(f"  Found {len(goalkeeper_tracks)} goalkeeper tracks after analysis: {sorted(goalkeeper_tracks.keys())}")
            for tid, info in goalkeeper_tracks.items():
                print(f"    Track {tid}: {info['goalkeeper_frame_count']}/{info['total_frames']} frames as GK ({info['goalkeeper_frame_count']/info['total_frames']*100:.1f}%)")
        else:
            print(f"  No goalkeeper tracks detected (all tracks are field players)")


        # Sort tracks by first appearance
        sorted_tracks = sorted(track_infos.items(), key=lambda x: x[1]['first_frame'] or float('inf'))

        # Find merge candidates
        merge_map = {}  # {fragment_id: main_id}
        used_ids = set()

        for i, (track_id_a, info_a) in enumerate(sorted_tracks):
            if track_id_a in merge_map or track_id_a in used_ids:
                continue  # Already merged or is a main track

            used_ids.add(track_id_a)

            # Look for tracks that might be continuations of this one
            for track_id_b, info_b in sorted_tracks[i+1:]:
                if track_id_b in merge_map or track_id_b in used_ids:
                    continue

                # CRITICAL: Never merge goalkeepers with field players
                if info_a['is_goalkeeper'] != info_b['is_goalkeeper']:
                    continue  # One is GK, one is field player - cannot be same person

                # Check if B could be continuation of A
                if info_a['last_frame'] is None or info_b['first_frame'] is None:
                    continue

                frame_gap = info_b['first_frame'] - info_a['last_frame']

                # Skip if gap too large or overlapping
                if frame_gap > self.max_gap_frames or frame_gap < 1:
                    continue

                # Check team match
                if info_a['team'] is not None and info_b['team'] is not None:
                    if info_a['team'] != info_b['team']:
                        continue  # Different teams, can't be same player

                # Check spatial proximity
                if info_a['last_position'] and info_b['first_position']:
                    distance = self._calculate_position_distance(
                        info_a['last_position'],
                        info_b['first_position']
                    )

                    if distance < self.max_position_distance:
                        # Merge B into A
                        merge_map[track_id_b] = track_id_a
                        # Update info_a to include B's end position for chain merging
                        info_a['last_frame'] = info_b['last_frame']
                        info_a['last_position'] = info_b['last_position']

        return merge_map

    def apply_merges(self, player_tracks: Dict, team_assignments: Dict,
                    merge_map: Dict[int, int]) -> Tuple[Dict, Dict]:
        """
        Apply track merges by reassigning ByteTrack IDs.

        Args:
            player_tracks: Original player tracks
            team_assignments: Original team assignments
            merge_map: Mapping of {fragment_id: main_id}

        Returns:
            Tuple of (merged_player_tracks, merged_team_assignments)
        """
        if not merge_map:
            return player_tracks, team_assignments

        merged_tracks = {}
        merged_teams = {}

        for frame_idx, frame_tracks in player_tracks.items():
            if -1 in frame_tracks:
                merged_tracks[frame_idx] = frame_tracks
                merged_teams[frame_idx] = team_assignments.get(frame_idx, {})
                continue

            merged_frame_tracks = {}
            merged_frame_teams = {}

            for bytetrack_id, bbox in frame_tracks.items():
                # Use main ID if this is a fragment
                main_id = merge_map.get(bytetrack_id, bytetrack_id)
                merged_frame_tracks[main_id] = bbox

                # Copy team assignment
                if frame_idx in team_assignments and bytetrack_id in team_assignments[frame_idx]:
                    merged_frame_teams[main_id] = team_assignments[frame_idx][bytetrack_id]

            merged_tracks[frame_idx] = merged_frame_tracks
            merged_teams[frame_idx] = merged_frame_teams

        return merged_tracks, merged_teams

        return merged_tracks, merged_teams
