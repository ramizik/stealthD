"""
Metrics Calculator module - orchestrates all analytics calculations.

Coordinates speed calculation, possession tracking, and pass detection
to generate comprehensive match analytics.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import gc
from typing import Dict, Tuple

import numpy as np
import supervision as sv

from .robust_pass_detector import RobustPassDetector
from .improved_player_ball_assigner import ImprovedPlayerBallAssigner
from .possession_tracker import PossessionTracker
from .speed_calculator import SpeedCalculator


class MetricsCalculator:
    """
    Orchestrator for all analytics calculations.

    Coordinates speed calculations, possession tracking, and pass detection
    to generate comprehensive player and team statistics.
    """

    def __init__(self, fps: float = 30, field_dimensions: Tuple[float, float] = (105, 68)):
        self.fps = fps
        self.field_dimensions = field_dimensions
        self.speed_calculator = SpeedCalculator(field_dimensions, fps)
        self.possession_tracker = PossessionTracker(
            possession_threshold=8.0,
            min_possession_frames=1
        )

        # Use ImprovedPlayerBallAssigner and RobustPassDetector (from user's working scripts)
        self.ball_assigner = ImprovedPlayerBallAssigner(
            max_ball_distance=10.0  # Distance in meters (default from user's code)
        )

        self.pass_detector = RobustPassDetector(
            min_pass_distance=0.3,
            max_pass_distance=70.0,
            max_gap_frames=150,
            velocity_threshold=0.5,
            min_ball_movement=2.0,
            enable_fallback_mode=True
        )

    def _calculate_ball_field_coordinates(self, ball_tracks: Dict, adaptive_mapper,
                                         video_dimensions: Tuple[int, int], camera_movement: list = None) -> Dict:
        """
        Calculate field coordinates for ball positions using adaptive per-frame homography.

        Args:
            ball_tracks: Dictionary {frame_idx: [x1, y1, x2, y2]} or {frame_idx: {0: {'bbox': [...], 'position_transformed': [...]}}}
            adaptive_mapper: AdaptiveHomographyMapper object
            video_dimensions: (width, height) of video in pixels
            camera_movement: List of [x_movement, y_movement] per frame

        Returns:
            Dictionary {frame_idx: [x, y]} with ball field coordinates
        """
        ball_field_coords = {}
        used_position_transformed = 0
        used_adaptive_mapper = 0
        failed_conversions = 0

        for frame_idx, track_data in ball_tracks.items():
            field_coord = None

            # Handle enhanced format: {0: {'bbox': [...], 'position_transformed': [...]}}
            if isinstance(track_data, dict) and 0 in track_data:
                track_info = track_data[0]
                if isinstance(track_info, dict):
                    # First, try to use position_transformed if available (from ViewTransformer)
                    position_transformed = track_info.get('position_transformed', None)
                    if position_transformed is not None:
                        # position_transformed is already in field coordinates (meters)
                        field_coord = np.array(position_transformed)
                        used_position_transformed += 1
                    else:
                        # Fall back to position_adjusted (pixel coordinates) with adaptive_mapper
                        position_adjusted = track_info.get('position_adjusted', None)
                        if position_adjusted is not None:
                            ball_center = np.array(position_adjusted)
                        else:
                            # Last resort: use bbox center
                            bbox = track_info.get('bbox', None)
                            if bbox is None or any(b is None for b in bbox) or len(bbox) < 4:
                                failed_conversions += 1
                                continue
                            x_center = (bbox[0] + bbox[2]) / 2
                            y_center = (bbox[1] + bbox[3]) / 2
                            ball_center = np.array([x_center, y_center])

                        # Get camera motion for this frame if available
                        camera_motion = None
                        if camera_movement and frame_idx < len(camera_movement):
                            camera_motion = tuple(camera_movement[frame_idx])

                        # Convert to field coordinates using adaptive transformation
                        field_coord, method, confidence = adaptive_mapper.transform_point(
                            frame_idx, ball_center, camera_motion
                        )
                        if field_coord is not None:
                            used_adaptive_mapper += 1
                        else:
                            failed_conversions += 1
                            continue
            # Handle simple format: list (bbox)
            elif isinstance(track_data, list) and len(track_data) >= 4:
                if any(b is None for b in track_data):
                    failed_conversions += 1
                    continue

                # Get center of ball bounding box
                x_center = (track_data[0] + track_data[2]) / 2
                y_center = (track_data[1] + track_data[3]) / 2
                ball_center = np.array([x_center, y_center])

                # Get camera motion for this frame if available
                camera_motion = None
                if camera_movement and frame_idx < len(camera_movement):
                    camera_motion = tuple(camera_movement[frame_idx])

                # Convert to field coordinates using adaptive transformation
                field_coord, method, confidence = adaptive_mapper.transform_point(
                    frame_idx, ball_center, camera_motion
                )
                if field_coord is not None:
                    used_adaptive_mapper += 1
                else:
                    failed_conversions += 1
                    continue
            else:
                failed_conversions += 1
                continue

            # Store the field coordinate
            if field_coord is not None:
                ball_field_coords[int(frame_idx)] = [float(field_coord[0]), float(field_coord[1])]

        # Debug output
        if len(ball_field_coords) == 0:
            print(f"      [ERROR] No ball field coordinates calculated!")
            print(f"        Used position_transformed: {used_position_transformed}")
            print(f"        Used adaptive_mapper: {used_adaptive_mapper}")
            print(f"        Failed conversions: {failed_conversions}")
        else:
            print(f"      [DEBUG] Ball coord sources: {used_position_transformed} from ViewTransformer, {used_adaptive_mapper} from adaptive_mapper")

        return ball_field_coords

    def calculate_all_metrics(self, all_tracks: Dict, adaptive_mapper,
                             video_info: sv.VideoInfo, camera_movement: list = None, id_mapping: Dict = None) -> Dict:
        """
        Calculate all analytics metrics from tracking data using adaptive homography.

        Args:
            all_tracks: Dictionary with keys 'player', 'ball', 'referee', 'player_classids'
            adaptive_mapper: AdaptiveHomographyMapper object
            video_info: Video information object from supervision
            camera_movement: List of [x_movement, y_movement] per frame

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
        print("Calculating analytics metrics with adaptive homography...")

        # Extract data from all_tracks
        player_tracks = all_tracks.get('player', {})
        ball_tracks = all_tracks.get('ball', {})
        team_assignments = all_tracks.get('player_classids', {})
        goalkeeper_assignments = all_tracks.get('player_is_goalkeeper', {})

        # Get video dimensions
        video_dimensions = (video_info.width, video_info.height)

        # Step 1: Calculate player speeds and field coordinates
        print("  - Calculating player speeds and field coordinates...")

        # Transform player_tracks from {frame: {player_id: bbox}} to {player_id: {frame: center}}
        player_positions_by_id = {}
        for frame_idx, frame_players in player_tracks.items():
            if not isinstance(frame_players, dict):
                continue
            for player_id, track_data in frame_players.items():
                if player_id == -1:
                    continue
                # Handle both formats: list (bbox) or dict (enhanced format)
                if isinstance(track_data, dict):
                    bbox = track_data.get('bbox', None)
                elif isinstance(track_data, list) and len(track_data) >= 4:
                    bbox = track_data
                else:
                    continue

                if bbox is None or len(bbox) < 4 or any(b is None for b in bbox):
                    continue

                if player_id not in player_positions_by_id:
                    player_positions_by_id[player_id] = {}
                # Calculate center of bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                player_positions_by_id[player_id][int(frame_idx)] = [center_x, center_y]

        print(f"  Transformed {len(player_tracks)} frames of tracking data for {len(player_positions_by_id)} players")

        player_speeds = self.speed_calculator.calculate_speeds(
            player_positions_by_id, adaptive_mapper, video_dimensions, camera_movement
        )

        # Clear memory after speed calculation
        gc.collect()

        # Step 2: Calculate ball field coordinates
        print("  - Calculating ball field coordinates...")
        print(f"    Ball tracks: {len(ball_tracks)} frames")
        ball_field_coords = self._calculate_ball_field_coordinates(
            ball_tracks, adaptive_mapper, video_dimensions, camera_movement
        )
        print(f"    Ball field coords calculated: {len(ball_field_coords)} frames")

        # Step 2.5: Build field_coords_players structure for ImprovedPlayerBallAssigner and RobustPassDetector
        # Structure: {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
        print("  - Building player field coordinates structure...")
        field_coords_players = {}
        total_player_coords = 0
        for player_id, positions in player_positions_by_id.items():
            field_coords_players[player_id] = {'field_coordinates': {}}
            for frame_idx, pixel_pos in positions.items():
                # Transform pixel position to field coordinates
                field_coord, method, confidence = adaptive_mapper.transform_point(
                    frame_idx, np.array(pixel_pos),
                    tuple(camera_movement[frame_idx]) if camera_movement and frame_idx < len(camera_movement) else None
                )
                if field_coord is not None:
                    field_coords_players[player_id]['field_coordinates'][frame_idx] = [float(field_coord[0]), float(field_coord[1])]
                    total_player_coords += 1

        print(f"    Built field coordinates for {len(field_coords_players)} players, {total_player_coords} total coordinate pairs")
        print(f"    Ball field coordinates: {len(ball_field_coords)} frames")

        # Clear memory after ball coordinate calculation
        gc.collect()

        # Step 3: Track ball possession
        print("  - Tracking ball possession...")
        possession_events, player_possession_time = self.possession_tracker.track_possessions(
            player_tracks, ball_tracks, player_speeds, ball_field_coords, self.fps
        )

        # Update possession events with team information
        possession_events = self.possession_tracker.update_possession_with_teams(
            possession_events, team_assignments
        )

        # Clear memory after possession tracking
        gc.collect()

        # Step 4: Frame-by-frame ball assignment for touch detection (before pass detection)
        print("  - Assigning ball to players frame-by-frame...")

        # Debug: Check data availability
        common_frames = set(player_tracks.keys()) & set(ball_tracks.keys())
        frames_with_player_coords = set()
        frames_with_ball_coords = set(ball_field_coords.keys())

        for player_id, player_data in field_coords_players.items():
            if 'field_coordinates' in player_data:
                frames_with_player_coords.update(player_data['field_coordinates'].keys())

        print(f"    Common frames (player_tracks & ball_tracks): {len(common_frames)}")
        print(f"    Frames with player field coords: {len(frames_with_player_coords)}")
        print(f"    Frames with ball field coords: {len(frames_with_ball_coords)}")
        print(f"    Overlap (player coords & ball coords): {len(frames_with_player_coords & frames_with_ball_coords)}")

        ball_assignments_with_distance = self.ball_assigner.assign_ball_for_all_frames(
            player_tracks, ball_tracks, field_coords_players, ball_field_coords
        )

        # Analyze assignment quality
        quality = self.ball_assigner.analyze_assignment_quality(ball_assignments_with_distance)
        print(f"    Ball assignment quality: avg_dist={quality['avg_distance']:.2f}m, "
            f"assignments={quality['total_assignments']}")

        # Get simple mapping for touch detection
        ball_assignments = self.ball_assigner.get_simple_assignments(ball_assignments_with_distance)

        # Count touches from frame-by-frame assignments
        player_touches = self.ball_assigner.count_player_touches(ball_assignments)
        touch_frames = self.ball_assigner.get_player_touch_frames(ball_assignments)

        # Clear memory after ball assignment
        gc.collect()

        # Step 5: Detect passes with RobustPassDetector
        print("  - Detecting passes with robust validation...")
        passes = self.pass_detector.detect_passes(
            possession_events, field_coords_players, team_assignments, self.fps,
            ball_field_coords=ball_field_coords,  # For trajectory validation
            ball_assignments=ball_assignments,     # For touch validation
            id_mapping=id_mapping                  # For ID conversion from ByteTrack to fixed IDs
        )

        # Count passes per player
        player_pass_counts = self.pass_detector.count_player_passes(passes)

        # Report validation statistics
        if passes:
            hybrid_validated = sum(1 for p in passes if p.get('validation_method') == 'hybrid')
            avg_confidence = sum(p.get('confidence_score', 0) for p in passes) / len(passes)
            print(f"    {len(passes)} passes detected ({hybrid_validated} hybrid-validated, avg confidence: {avg_confidence:.2f})")

        # Clear memory after pass detection
        gc.collect()

        # Step 7: Aggregate player analytics
        print("  - Aggregating player analytics...")
        player_analytics = self._aggregate_player_analytics(
            player_speeds, player_possession_time, player_pass_counts, team_assignments,
            player_touches, goalkeeper_assignments
        )

        print(f"Analytics complete: {len(player_analytics)} players, {len(passes)} passes detected, {sum(player_touches.values())} total touches")

        # Final cleanup
        gc.collect()

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
                                    player_pass_counts: Dict, team_assignments: Dict, player_touches: Dict = None,
                                    goalkeeper_assignments: Dict = None) -> Dict:
        """
        Aggregate all analytics data per player.

        Args:
            player_speeds: Speed data from SpeedCalculator
            player_possession_time: Possession time from PossessionTracker
            player_pass_counts: Pass counts from PassDetector
            team_assignments: Team assignments {frame_idx: {player_id: team_id}}
            player_touches: Touch counts from PlayerBallAssigner (optional)
            goalkeeper_assignments: Goalkeeper status {frame_idx: {player_id: is_goalkeeper}} (optional)

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

            # Get goalkeeper status (check if player was ever marked as goalkeeper)
            analytics['is_goalkeeper'] = self._is_player_goalkeeper(player_id, goalkeeper_assignments)

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

    def _is_player_goalkeeper(self, player_id: int, goalkeeper_assignments: Dict) -> bool:
        """
        Check if a player was ever marked as a goalkeeper.

        Args:
            player_id: Player ID
            goalkeeper_assignments: Dictionary {frame_idx: {player_id: is_goalkeeper}}

        Returns:
            True if player was marked as goalkeeper in any frame, False otherwise
        """
        if not goalkeeper_assignments:
            return False

        for frame_idx, frame_goalkeepers in goalkeeper_assignments.items():
            if player_id in frame_goalkeepers and frame_goalkeepers[player_id]:
                return True

        return False

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
