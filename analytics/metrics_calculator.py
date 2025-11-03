"""
Metrics Calculator module - orchestrates all analytics calculations.

Coordinates speed calculation, possession tracking, and pass detection
to generate comprehensive match analytics.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, Tuple
import supervision as sv

from .speed_calculator import SpeedCalculator
from .possession_tracker import PossessionTracker
from .pass_detector import PassDetector
from .player_ball_assigner import PlayerBallAssigner


class MetricsCalculator:
    """
    Orchestrator for all analytics calculations.

    Coordinates speed calculations, possession tracking, and pass detection
    to generate comprehensive player and team statistics.
    """

    def __init__(self, fps: float = 30, field_dimensions: Tuple[float, float] = (105, 68)):
        """
        Initialize the metrics calculator.

        Args:
            fps: Frames per second of the video
            field_dimensions: (width, height) in meters. Default: (105, 68)
        """
        self.fps = fps
        self.field_dimensions = field_dimensions
        self.speed_calculator = SpeedCalculator(field_dimensions, fps)
        # Enhanced possession tracker with aligned parameters
        self.possession_tracker = PossessionTracker(
            possession_threshold=2.0,  # Match max_ball_distance for consistent detection
            min_possession_frames=1  # Capture even brief 1-frame touches (one-touch passes)
        )
        # Enhanced pass detector with hybrid validation
        self.pass_detector = PassDetector(
            min_pass_distance=1.0,       # Allow short passes (reduced from 2.5m)
            min_possession_duration=0.0, # No minimum duration - allow instant one-touch passes
            max_gap_frames=100,          # Allow up to 4 seconds ball flight time (includes loose balls)
            max_pass_distance=40.0,      # Ignore unrealistic long distances
            velocity_threshold=2.0       # Minimum ball speed for valid passes (m/s)
        )
        self.ball_assigner = PlayerBallAssigner(
            max_ball_distance=2.0  # Reduced from 2.5m for more accurate touch detection
        )

    def _calculate_ball_field_coordinates(self, ball_tracks: Dict, global_mapper,
                                         video_dimensions: Tuple[int, int]) -> Dict:
        """
        Calculate field coordinates for ball positions using global homography.

        Args:
            ball_tracks: Dictionary {frame_idx: [x1, y1, x2, y2]}
            global_mapper: GlobalHomographyMapper object
            video_dimensions: (width, height) of video in pixels

        Returns:
            Dictionary {frame_idx: [x, y]} with ball field coordinates
        """
        ball_field_coords = {}

        for frame_idx, bbox in ball_tracks.items():
            if bbox is None or any(b is None for b in bbox):
                continue

            # Get center of ball bounding box
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            ball_center = np.array([[x_center, y_center]])

            # Convert to field coordinates using global homography
            field_coord, _ = self.speed_calculator.calculate_field_coordinates(
                ball_center, global_mapper, video_dimensions, frame_idx
            )

            # Skip if conversion failed or is low confidence
            if field_coord is not None and len(field_coord) > 0:
                ball_field_coords[int(frame_idx)] = [float(field_coord[0][0]), float(field_coord[0][1])]

        return ball_field_coords

    def calculate_all_metrics(self, all_tracks: Dict, global_mapper,
                             video_info: sv.VideoInfo) -> Dict:
        """
        Calculate all analytics metrics from tracking data using global homography.

        Args:
            all_tracks: Dictionary with keys 'player', 'ball', 'referee', 'player_classids'
            global_mapper: GlobalHomographyMapper object
            video_info: Video information object from supervision

        Returns:
            Dictionary containing all calculated metrics:
            {
                'fps': float,
                'player_speeds': {...},
                'possession_data': {...},
                'passes': [...],
                'ball_assignments': {frame_idx: player_id},
                'player_touches': {player_id: touch_count},
                'touch_frames': {player_id: [frame_indices]},
                'player_analytics': {...}
            }
        """
        print("Calculating analytics metrics with global homography...")

        # Extract data from all_tracks
        player_tracks = all_tracks.get('player', {})
        ball_tracks = all_tracks.get('ball', {})
        team_assignments = all_tracks.get('player_classids', {})

        # Get video dimensions
        video_dimensions = (video_info.width, video_info.height)

        # Step 1: Calculate player speeds and field coordinates
        print("  - Calculating player speeds and field coordinates...")

        player_speeds = self.speed_calculator.calculate_speeds(
            player_tracks, global_mapper, video_dimensions
        )

        # Step 2: Calculate ball field coordinates
        print("  - Calculating ball field coordinates...")
        ball_field_coords = self._calculate_ball_field_coordinates(
            ball_tracks, global_mapper, video_dimensions
        )

        # Step 3: Track ball possession
        print("  - Tracking ball possession...")
        possession_events, player_possession_time = self.possession_tracker.track_possessions(
            player_tracks, ball_tracks, player_speeds, ball_field_coords, self.fps
        )

        # Update possession events with team information
        possession_events = self.possession_tracker.update_possession_with_teams(
            possession_events, team_assignments
        )

        # Step 4: Frame-by-frame ball assignment for touch detection (before pass detection)
        print("  - Assigning ball to players frame-by-frame...")
        ball_assignments = self.ball_assigner.assign_ball_for_all_frames(
            player_tracks, ball_tracks, player_speeds, ball_field_coords
        )

        # Count touches from frame-by-frame assignments
        player_touches = self.ball_assigner.count_player_touches(ball_assignments)
        touch_frames = self.ball_assigner.get_player_touch_frames(ball_assignments)

        # Step 5: Detect passes with HYBRID VALIDATION (possession + touches + velocity)
        print("  - Detecting passes with hybrid validation...")
        passes = self.pass_detector.detect_passes(
            possession_events, player_speeds, team_assignments, self.fps,
            ball_field_coords=ball_field_coords,  # For velocity validation
            ball_assignments=ball_assignments      # For touch validation
        )

        # Count passes per player
        player_pass_counts = self.pass_detector.count_player_passes(passes)

        # Report validation statistics
        if passes:
            hybrid_validated = sum(1 for p in passes if p.get('validation_method') == 'hybrid')
            avg_confidence = sum(p.get('confidence_score', 0) for p in passes) / len(passes)
            print(f"    {len(passes)} passes detected ({hybrid_validated} hybrid-validated, avg confidence: {avg_confidence:.2f})")

        # Step 7: Aggregate player analytics
        print("  - Aggregating player analytics...")
        player_analytics = self._aggregate_player_analytics(
            player_speeds, player_possession_time, player_pass_counts, team_assignments, player_touches
        )

        print(f"Analytics complete: {len(player_analytics)} players, {len(passes)} passes detected, {sum(player_touches.values())} total touches")

        return {
            'fps': self.fps,
            'player_speeds': player_speeds,
            'possession_data': {
                'possession_events': possession_events,
                'player_possession_time': player_possession_time
            },
            'passes': passes,
            'ball_assignments': ball_assignments,
            'player_touches': player_touches,
            'touch_frames': touch_frames,
            'player_analytics': player_analytics
        }

    def _aggregate_player_analytics(self, player_speeds: Dict, player_possession_time: Dict,
                                    player_pass_counts: Dict, team_assignments: Dict, player_touches: Dict = None) -> Dict:
        """
        Aggregate all analytics data per player.

        Args:
            player_speeds: Speed data from SpeedCalculator
            player_possession_time: Possession time from PossessionTracker
            player_pass_counts: Pass counts from PassDetector
            team_assignments: Team assignments {frame_idx: {player_id: team_id}}
            player_touches: Touch counts from PlayerBallAssigner (optional)

        Returns:
            Dictionary {player_id: {all analytics combined}}
        """
        player_analytics = {}

        # Get all unique player IDs
        all_player_ids = set(player_speeds.keys())
        all_player_ids.update(player_possession_time.keys())
        all_player_ids.update(player_pass_counts.keys())
        if player_touches:
            all_player_ids.update(player_touches.keys())

        for player_id in all_player_ids:
            analytics = {}

            # Get team assignment (use most common team for this player)
            player_team = self._get_player_team(player_id, team_assignments)
            if player_team is not None:
                analytics['team'] = int(player_team)

            # Speed metrics
            if player_id in player_speeds:
                speed_data = player_speeds[player_id]
                analytics['average_speed_kmh'] = speed_data.get('average_speed_kmh', 0.0)
                analytics['max_speed_kmh'] = speed_data.get('max_speed_kmh', 0.0)
                analytics['total_distance_m'] = speed_data.get('total_distance_m', 0.0)
            else:
                analytics['average_speed_kmh'] = 0.0
                analytics['max_speed_kmh'] = 0.0
                analytics['total_distance_m'] = 0.0

            # Possession metrics
            analytics['ball_possession_seconds'] = player_possession_time.get(player_id, 0.0)

            # Touch count (frame-by-frame ball assignment)
            analytics['ball_touches'] = player_touches.get(player_id, 0) if player_touches else 0

            # Pass metrics
            if player_id in player_pass_counts:
                analytics['passes_completed'] = player_pass_counts[player_id].get('passes_completed', 0)
                analytics['passes_received'] = player_pass_counts[player_id].get('passes_received', 0)
            else:
                analytics['passes_completed'] = 0
                analytics['passes_received'] = 0

            player_analytics[int(player_id)] = analytics

        return player_analytics

    def _get_player_team(self, player_id: int, team_assignments: Dict) -> int:
        """
        Get the most common team assignment for a player.

        Args:
            player_id: Player ID
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}

        Returns:
            Team ID (0 or 1) or None if no assignment found
        """
        team_votes = []

        for frame_idx, frame_teams in team_assignments.items():
            if player_id in frame_teams:
                team_votes.append(frame_teams[player_id])

        if not team_votes:
            return None

        # Return most common team
        return max(set(team_votes), key=team_votes.count)
