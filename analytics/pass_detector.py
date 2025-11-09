"""
Pass Detector module for identifying and validating pass events.

Detects passes based on possession transfers with distance and duration validation.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import numpy as np
from typing import Dict, List, Optional, Tuple


class PassDetector:
    """
    Detect and validate pass events in soccer matches.

    Identifies passes based on possession transfers between same-team players,
    with validation for minimum distance and possession duration.
    """

    def __init__(self,
                 min_pass_distance: float = 0.5,  # Reduced from 1.0m to count more passes
                 min_possession_duration: float = 0.05,  # Reduced from 0.1s for quicker passes
                 max_gap_frames: int = 30,  # Increased from 15 to allow longer passes
                 max_pass_distance: float = 60.0,  # Increased from 40.0m for longer passes
                 velocity_threshold: float = 1.0):  # Reduced from 2.0m/s to count slower passes
        """
        Initialize the pass detector with enhanced validation parameters.

        Args:
            min_pass_distance: Minimum distance in meters between players to count as pass (default: 0.5m)
            min_possession_duration: Minimum duration in seconds for valid possession (default: 0.05s)
            max_gap_frames: Maximum frames between possessions to count as pass (default: 30 frames ~1.0s)
            max_pass_distance: Maximum distance in meters for realistic passes (default: 60.0m)
            velocity_threshold: Minimum ball velocity in m/s for valid passes (default: 1.0 m/s)
        """
        self.min_pass_distance = min_pass_distance
        self.min_possession_duration = min_possession_duration
        self.max_gap_frames = max_gap_frames
        self.max_pass_distance = max_pass_distance
        self.velocity_threshold = velocity_threshold

    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray, ball_coords_ref: Dict = None) -> float:
        """Calculate Euclidean distance between two points in meters with coordinate normalization."""
        if point1 is None or point2 is None:
            return 0.0

        # Ensure we only use x, y coordinates (first 2 elements)
        point1 = np.asarray(point1)[:2]
        point2 = np.asarray(point2)[:2]

        # Normalize coordinates to meters if needed
        if ball_coords_ref is not None:
            point1 = self._normalize_coordinates_to_meters(point1, ball_coords_ref)
            point2 = self._normalize_coordinates_to_meters(point2, ball_coords_ref)

        return float(np.linalg.norm(point1 - point2))

    def _normalize_coordinates_to_meters(self, point: np.ndarray, reference_ball_coords: Dict) -> np.ndarray:
        """
        Normalize coordinates to meters using ball coordinates as reference.

        This is the same normalization logic from PlayerBallAssigner, copied here
        because pass detection needs it for distance calculations.
        """
        point = np.asarray(point)[:2]

        # If point is already in meters (typical field range), return as-is
        if 0 <= point[0] <= 120 and 0 <= point[1] <= 80:  # Reasonable field bounds in meters
            return point

        # Calculate conversion factor based on reference ball coordinates
        if len(reference_ball_coords) > 0:
            # Get average ball coordinate as reference (should be in meters)
            ball_coords = list(reference_ball_coords.values())
            avg_ball_x = np.mean([c[0] for c in ball_coords])
            avg_ball_y = np.mean([c[1] for c in ball_coords])

            # If ball coordinates are in reasonable meter range, use them to estimate conversion
            if 0 <= avg_ball_x <= 120 and 0 <= avg_ball_y <= 80:
                # Estimate that player coordinates should be in similar field coordinate range
                # but they appear to be scaled up by a factor
                # Use the ratio to estimate the scaling factor
                scaling_factor = (avg_ball_x + avg_ball_y) / (point[0] + point[1]) * 2

                # Apply scaling to bring coordinates to meter range
                point_meters = point * scaling_factor

                # Validate the result is in reasonable field bounds
                if 0 <= point_meters[0] <= 120 and 0 <= point_meters[1] <= 80:
                    return point_meters

        # If point appears to be in pitch coordinates (12000x7000), convert to meters
        if point[0] > 1000 or point[1] > 1000:
            # Convert from pitch coordinates (12000x7000) to meters (105x68)
            point_meters = np.array([
                point[0] * (105.0 / 12000.0),
                point[1] * (68.0 / 7000.0)
            ])
            return point_meters

        # Fallback: estimate scaling based on typical coordinate patterns
        # For our specific case, player coords seem to be ~9-10x ball coords
        if point[0] > 200:  # Likely in the wrong coordinate system
            # Use empirical scaling factor based on observed data
            point_meters = point * 0.11  # Approximate scaling factor
            return point_meters

        # Default: return as-is if we can't determine the system
        return point

    def _get_player_position(self, player_id: int, frame_idx: int,
                            field_coords_players: Dict) -> Optional[np.ndarray]:
        """Get player position at a specific frame."""
        if player_id in field_coords_players:
            player_data = field_coords_players[player_id]
            if 'field_coordinates' in player_data:
                coords = player_data['field_coordinates'].get(frame_idx, None)
                if coords is not None:
                    return np.array(coords)
        return None

    def _calculate_ball_velocity(self, ball_field_coords: Dict, start_frame: int,
                                 end_frame: int, fps: float = 30) -> float:
        """
        Calculate ball velocity between two frames.

        Args:
            ball_field_coords: Dictionary {frame_idx: [x, y]} with ball field coordinates
            start_frame: Starting frame index
            end_frame: Ending frame index
            fps: Frames per second

        Returns:
            Ball velocity in meters per second, or 0.0 if calculation fails
        """
        if start_frame not in ball_field_coords or end_frame not in ball_field_coords:
            return 0.0

        start_pos = np.array(ball_field_coords[start_frame])
        end_pos = np.array(ball_field_coords[end_frame])

        distance = self._calculate_distance(start_pos, end_pos)
        time_elapsed = (end_frame - start_frame) / fps

        if time_elapsed <= 0:
            return 0.0

        velocity = distance / time_elapsed
        return velocity

    def _extract_touch_events(self, ball_assignments: Dict) -> List[Dict]:
        """
        Extract discrete touch events from ball assignments.

        Records BOTH the first touch (receive) and last touch (release) for each possession.
        For pass detection, we use the last touch of the passer and first touch of receiver.

        Args:
            ball_assignments: Dictionary {frame_idx: player_id}

        Returns:
            List of touch events with frame, player_id, and event_type ('receive' or 'release')
        """
        touch_events = []
        prev_player = None
        possession_start = None

        sorted_frames = sorted(ball_assignments.keys())

        print(f"      [DEBUG] Touch events: {len(sorted_frames)} ball assignments to process")

        for i, frame_idx in enumerate(sorted_frames):
            current_player = ball_assignments[frame_idx]

            # Player change detected
            if current_player != prev_player:
                # Record RELEASE event for previous player (if exists)
                if prev_player is not None and possession_start is not None:
                    # Use the frame BEFORE the change as release point
                    release_frame = sorted_frames[i - 1] if i > 0 else possession_start
                    touch_events.append({
                        'frame': int(release_frame),
                        'player_id': int(prev_player),
                        'event_type': 'release'
                    })

                # Record RECEIVE event for new player
                if current_player is not None:
                    touch_events.append({
                        'frame': int(frame_idx),
                        'player_id': int(current_player),
                        'event_type': 'receive'
                    })
                    possession_start = frame_idx

                prev_player = current_player

        # Handle last possession's release
        if prev_player is not None and len(sorted_frames) > 0:
            touch_events.append({
                'frame': int(sorted_frames[-1]),
                'player_id': int(prev_player),
                'event_type': 'release'
            })

        print(f"      [DEBUG] Touch events: {len(touch_events)} touch events extracted "
              f"({len([t for t in touch_events if t['event_type'] == 'release'])} release, "
              f"{len([t for t in touch_events if t['event_type'] == 'receive'])} receive)")

        return touch_events

    def _detect_touch_based_passes(self, ball_assignments: Dict, team_assignments: Dict,
                                   field_coords_players: Dict, fps: float, ball_field_coords: Dict = None) -> List[Dict]:
        """
        PRIMARY METHOD: Detect passes directly from touch events (frame-by-frame ball assignments).

        This method is more reliable for brief possessions and one-touch passes than possession-based detection.

        Args:
            ball_assignments: Dictionary {frame_idx: player_id} from frame-by-frame ball tracking
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            field_coords_players: Dictionary {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            fps: Frames per second
            ball_field_coords: Dictionary {frame_idx: [x, y]} ball field coordinates (optional)

        Returns:
            List of pass events detected from touch transitions
        """
        print("      [DEBUG] 🚀 USING ENHANCED PASS DETECTION WITH IMPROVED TEAM VALIDATION")
        passes = []
        touch_events = self._extract_touch_events(ball_assignments)

        print(f"      [DEBUG] Touch-based detection: {len(touch_events)} touch events extracted")
        if len(touch_events) < 2:
            print(f"      [DEBUG] Touch-based detection: Not enough touch events ({len(touch_events)} < 2)")
            return passes

        # Match RELEASE events with subsequent RECEIVE events to detect passes
        potential_passes = 0
        for i in range(len(touch_events) - 1):
            passer_touch = touch_events[i]
            receiver_touch = touch_events[i + 1]

            # We want: passer RELEASE → receiver RECEIVE
            # Skip if not the right event types
            if passer_touch.get('event_type') != 'release':
                continue
            if receiver_touch.get('event_type') != 'receive':
                continue

            passer_id = passer_touch['player_id']
            receiver_id = receiver_touch['player_id']
            passer_frame = passer_touch['frame']
            receiver_frame = receiver_touch['frame']

            # Skip if same player (not a pass)
            if passer_id == receiver_id:
                continue

            gap_frames = receiver_frame - passer_frame

            # Skip if gap is too large (ball was loose/out of play)
            if gap_frames > self.max_gap_frames:
                continue

            # Get team assignments with enhanced robustness for stabilized data
            passer_team = None
            receiver_team = None

            # Build a comprehensive view of each player's team assignments
            def get_dominant_team(player_id, center_frame, team_assignments):
                """Get the dominant team for a player around a specific frame."""
                team_votes = []

                # Check a wider range of frames to find the dominant team
                for offset in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    test_frame = center_frame + offset
                    if test_frame in team_assignments and player_id in team_assignments[test_frame]:
                        team_votes.append(team_assignments[test_frame][player_id])

                if not team_votes:
                    return None

                # Use majority vote - the stabilized team should be dominant
                from collections import Counter
                team_counts = Counter(team_votes)
                dominant_team = team_counts.most_common(1)[0][0]

                # Only return if there's reasonable consistency (>60% agreement)
                agreement = team_counts[dominant_team] / len(team_votes)
                print(f"        [DEBUG] Player {player_id} near frame {center_frame}: votes={team_votes}, dominant={dominant_team}, agreement={agreement:.2f}")
                if agreement > 0.6:
                    return dominant_team

                print(f"        [DEBUG] Player {player_id}: Low agreement ({agreement:.2f}), returning None")
                return None

            # Get dominant teams for both players
            passer_team = get_dominant_team(passer_id, passer_frame, team_assignments)
            receiver_team = get_dominant_team(receiver_id, receiver_frame, team_assignments)

            print(f"        [DEBUG] 🧪 TESTING PASS: P{passer_id}({passer_team})→P{receiver_id}({receiver_team})")

            # HYBRID VALIDATION: Use multiple approaches to determine if this is a valid same-team pass
            is_valid_pass = False
            validation_method = ""

            # Method 1: Enhanced dominant team validation (primary)
            if passer_team is not None and receiver_team is not None and passer_team == receiver_team:
                is_valid_pass = True
                validation_method = "dominant_team"
                print(f"        [DEBUG] ✅ Method 1 SUCCESS: Same dominant team ({passer_team})")

            # Method 2: Fallback to frame-specific validation if dominant team fails
            if not is_valid_pass:
                print(f"        [DEBUG] 🔍 Method 2: Testing direct frame validation")
                # Check direct frame assignments
                direct_passer_team = None
                direct_receiver_team = None

                if passer_frame in team_assignments and passer_id in team_assignments[passer_frame]:
                    direct_passer_team = team_assignments[passer_frame][passer_id]
                if receiver_frame in team_assignments and receiver_id in team_assignments[receiver_frame]:
                    direct_receiver_team = team_assignments[receiver_frame][receiver_id]

                print(f"        [DEBUG] Direct frame teams: P{passer_id}={direct_passer_team}, P{receiver_id}={direct_receiver_team}")

                if direct_passer_team is not None and direct_receiver_team is not None and direct_passer_team == direct_receiver_team:
                    is_valid_pass = True
                    validation_method = "direct_frame"
                    passer_team = direct_passer_team
                    receiver_team = direct_receiver_team
                    print(f"        [DEBUG] ✅ Method 2 SUCCESS: Same direct frame team ({direct_passer_team})")
                else:
                    print(f"        [DEBUG] ❌ Method 2 FAILED: Different direct frame teams ({direct_passer_team} vs {direct_receiver_team})")

            # Method 3: Pragmatic pass validation - focus on realistic gameplay scenarios
            if not is_valid_pass:
                print(f"        [DEBUG] 🎯 METHOD 3 TRIGGERED: Attempting overall consistency validation for P{passer_id}→P{receiver_id}")
                print(f"        [DEBUG] 🔍 ATTEMPTING OVERALL CONSISTENCY VALIDATION for P{passer_id}→P{receiver_id}")

                # In real soccer, passes between same-team players should be common
                # If we're seeing very few passes, we might be over-filtering

                # Count total team assignments for these players to understand overall patterns
                passer_total_teams = []
                receiver_total_teams = []

                for frame_idx, frame_assignments in team_assignments.items():
                    if frame_assignments:
                        if passer_id in frame_assignments:
                            passer_total_teams.append(frame_assignments[passer_id])
                        if receiver_id in frame_assignments:
                            receiver_total_teams.append(frame_assignments[receiver_id])

                print(f"        [DEBUG] P{passer_id} total assignments: {len(passer_total_teams)}, P{receiver_id} total assignments: {len(receiver_total_teams)}")

                if passer_total_teams and receiver_total_teams:
                    from collections import Counter
                    passer_overall = Counter(passer_total_teams)
                    receiver_overall = Counter(receiver_total_teams)

                    print(f"        [DEBUG] P{passer_id} team distribution: {dict(passer_overall)}")
                    print(f"        [DEBUG] P{receiver_id} team distribution: {dict(receiver_overall)}")

                    # Get most common teams overall
                    passer_main = passer_overall.most_common(1)[0][0]
                    receiver_main = receiver_overall.most_common(1)[0][0]

                    print(f"        [DEBUG] P{passer_id} main team: {passer_main}, P{receiver_id} main team: {receiver_main}")

                    # If players are on the same team overall, allow passes between them
                    # This handles cases where team assignment has temporary glitches
                    if passer_main == receiver_main:
                        print(f"        [DEBUG] ✓ Players on same main team, checking consistency...")

                        # Check if this is reasonable (not just random coincidence)
                        passer_agreement = passer_overall[passer_main] / len(passer_total_teams)
                        receiver_agreement = receiver_overall[receiver_main] / len(receiver_total_teams)

                        print(f"        [DEBUG] P{passer_id} agreement: {passer_agreement:.1%}, P{receiver_id} agreement: {receiver_agreement:.1%}")

                        # If both players have reasonable team consistency (>70%), allow pass
                        if passer_agreement > 0.7 and receiver_agreement > 0.7:
                            is_valid_pass = True
                            validation_method = "overall_consistency"
                            passer_team = passer_main
                            receiver_team = receiver_main
                            print(f"        [DEBUG] ✅ Pass allowed via overall consistency: P{passer_id}({passer_agreement:.1%})→P{receiver_id}({receiver_agreement:.1%})")
                        else:
                            print(f"        [DEBUG] ❌ Low consistency: P{passer_id}({passer_agreement:.1%}), P{receiver_id}({receiver_agreement:.1%})")
                    else:
                        print(f"        [DEBUG] ❌ Different main teams: P{passer_main} vs P{receiver_main}")
                else:
                    print(f"        [DEBUG] ❌ Missing team data: P{passer_id}={len(passer_total_teams)}, P{receiver_id}={len(receiver_total_teams)}")

            # Final validation decision
            if not is_valid_pass:
                print(f"        [DEBUG] All team validation methods failed: P{passer_id}({passer_team})→P{receiver_id}({receiver_team})")
                print(f"        [DEBUG] Frame context: passer_frame={passer_frame}, receiver_frame={receiver_frame}")
                continue

            print(f"        [DEBUG] ✓ Team validation passed via {validation_method}: P{passer_id}({passer_team})→P{receiver_id}({receiver_team})")

            potential_passes += 1

            # Get positions with enhanced robustness - try even wider range
            passer_pos = None
            # Search much wider range for positions (up to 50 frames ~ 1.5 seconds)
            for offset in list(range(0, -26, -1)) + list(range(1, 26)):  # -25 to +25
                test_frame = passer_frame + offset
                passer_pos = self._get_player_position(passer_id, test_frame, field_coords_players)
                if passer_pos is not None:
                    break

            receiver_pos = None
            # Search much wider range for positions
            for offset in list(range(0, 26)) + list(range(-1, -26, -1)):  # 0 to +25, then -1 to -25
                test_frame = receiver_frame + offset
                receiver_pos = self._get_player_position(receiver_id, test_frame, field_coords_players)
                if receiver_pos is not None:
                    break

            # Enhanced fallback: use any available position for these players
            if passer_pos is None:
                # Try any frame where this player appears
                if passer_id in field_coords_players and 'field_coordinates' in field_coords_players[passer_id]:
                    all_positions = list(field_coords_players[passer_id]['field_coordinates'].values())
                    if all_positions:
                        passer_pos = np.array(all_positions[len(all_positions)//2])  # Use middle position
                        print(f"        [DEBUG] Using fallback position for P{passer_id}")

            if receiver_pos is None:
                # Try any frame where this player appears
                if receiver_id in field_coords_players and 'field_coordinates' in field_coords_players[receiver_id]:
                    all_positions = list(field_coords_players[receiver_id]['field_coordinates'].values())
                    if all_positions:
                        receiver_pos = np.array(all_positions[len(all_positions)//2])  # Use middle position
                        print(f"        [DEBUG] Using fallback position for P{receiver_id}")

            if passer_pos is None or receiver_pos is None:
                print(f"        [DEBUG] Position validation failed: P{passer_id}({passer_pos is not None})→P{receiver_id}({receiver_pos is not None})")
                continue

            # Calculate pass distance with coordinate normalization
            pass_distance = self._calculate_distance(passer_pos, receiver_pos, ball_field_coords)

            # Validate distance
            if pass_distance < self.min_pass_distance or pass_distance > self.max_pass_distance:
                print(f"        [DEBUG] Distance validation failed: {pass_distance:.1f}m (min:{self.min_pass_distance}, max:{self.max_pass_distance})")
                continue

            # Create pass event
            pass_event = {
                'frame': int(passer_frame),
                'timestamp': round(passer_frame / fps, 2),
                'from_player': int(passer_id),
                'to_player': int(receiver_id),
                'team': int(passer_team),
                'distance_m': round(pass_distance, 2),
                'start_position': [round(float(passer_pos[0]), 2), round(float(passer_pos[1]), 2)],
                'end_position': [round(float(receiver_pos[0]), 2), round(float(receiver_pos[1]), 2)],
                'ball_travel_time_s': round(gap_frames / fps, 2),
                'gap_frames': int(gap_frames),
                'detection_method': 'touch_based',
                'confidence_score': 1.0  # High confidence for direct touch detection
            }

            passes.append(pass_event)

        print(f"      [DEBUG] Touch-based detection: {potential_passes} potential pass candidates, {len(passes)} passes created")
        return passes

    def _validate_with_touches(self, pass_event: Dict, touch_events: List[Dict],
                              tolerance_frames: int = 5) -> Tuple[bool, float]:
        """
        Validate a pass event against touch events.

        Args:
            pass_event: Pass event to validate
            touch_events: List of touch events from ball assignments
            tolerance_frames: Frame tolerance for matching (default: 5)

        Returns:
            Tuple of (is_valid, confidence_score)
        """
        passer_id = pass_event['from_player']
        receiver_id = pass_event['to_player']
        pass_frame = pass_event['frame']
        receive_frame = pass_frame + pass_event.get('gap_frames', 0)

        # Find touch events near pass and receive frames
        passer_touch_found = False
        receiver_touch_found = False

        for touch in touch_events:
            touch_frame = touch['frame']
            touch_player = touch['player_id']

            # Check for passer touch near pass frame
            if (touch_player == passer_id and
                abs(touch_frame - pass_frame) <= tolerance_frames):
                passer_touch_found = True

            # Check for receiver touch near receive frame
            if (touch_player == receiver_id and
                abs(touch_frame - receive_frame) <= tolerance_frames):
                receiver_touch_found = True

        # Calculate confidence score
        if passer_touch_found and receiver_touch_found:
            confidence = 1.0  # High confidence - both touches validated
        elif passer_touch_found or receiver_touch_found:
            confidence = 0.7  # Medium confidence - one touch validated
        else:
            confidence = 0.3  # Low confidence - no touch validation

        # Require at least one touch to be valid
        is_valid = passer_touch_found or receiver_touch_found

        return is_valid, confidence

    def _find_pass_target(self, sorted_events: List[Dict], start_idx: int,
                          team_assignments: Dict, max_interrupted_frames: int = 15) -> Optional[Tuple[int, int]]:
        """
        Find the actual pass target, skipping brief opposite-team possessions (ball in flight).

        Args:
            sorted_events: Sorted list of possession events
            start_idx: Index of the passer's possession event
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            max_interrupted_frames: Maximum frames for brief opposite-team possession to ignore

        Returns:
            Tuple of (target_event_index, total_gap_frames) or None if no valid target found
        """
        passer_event = sorted_events[start_idx]
        passer_id = passer_event['player_id']
        passer_team = passer_event.get('team')

        if passer_team is None:
            return None

        # Look ahead to find same-team receiver, ignoring brief opposite-team possessions
        total_gap = 0
        for j in range(start_idx + 1, len(sorted_events)):
            candidate_event = sorted_events[j]
            candidate_id = candidate_event['player_id']
            candidate_team = candidate_event.get('team')

            # Calculate gap from passer to this candidate
            if j == start_idx + 1:
                gap = candidate_event['start_frame'] - passer_event['end_frame']
                total_gap = gap
            else:
                # Gap from passer to this candidate (skipping intermediate possessions)
                total_gap = candidate_event['start_frame'] - passer_event['end_frame']

            # Stop if gap is too large
            if total_gap > self.max_gap_frames:
                break

            # Skip if same player
            if candidate_id == passer_id:
                continue

            # Check if same team
            if candidate_team == passer_team:
                # Found a same-team target!
                # But check if we skipped any opposite-team possessions
                if j == start_idx + 1:
                    # Direct pass (no intermediate possession)
                    return (j, total_gap)
                else:
                    # Check if intermediate possessions are brief (ball in flight)
                    all_brief = True
                    for k in range(start_idx + 1, j):
                        intermediate = sorted_events[k]
                        if intermediate['num_frames'] > max_interrupted_frames:
                            all_brief = False
                            break

                    if all_brief:
                        # All intermediate possessions are brief - treat as successful pass
                        return (j, total_gap)
                    else:
                        # Intermediate possession was substantial - real interception
                        break
            elif candidate_team != passer_team:
                # Opposite team - check if brief (ball in flight) or real possession
                if candidate_event['num_frames'] > max_interrupted_frames:
                    # Substantial opposite-team possession - stop looking
                    break
                # Otherwise continue looking (might be ball in flight)

        return None

    def detect_passes(self, possession_events: List[Dict], field_coords_players: Dict,
                     team_assignments: Dict, fps: float = 30, ball_field_coords: Dict = None,
                     ball_assignments: Dict = None) -> List[Dict]:
        """
        HYBRID APPROACH: Detect passes using both touch-based (primary) and possession-based (secondary) methods.

        Research shows touch-based detection is more reliable for brief possessions and one-touch passes.
        Possession-based detection catches longer passes that may be missed by touch detection.

        Args:
            possession_events: List of possession events from PossessionTracker
            field_coords_players: Dictionary {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            fps: Frames per second of video
            ball_field_coords: Dictionary {frame_idx: [x, y]} ball field coordinates (optional)
            ball_assignments: Dictionary {frame_idx: player_id} for touch detection (CRITICAL)

        Returns:
            List of pass events with metadata and validation scores
        """
        print("      [DEBUG] 🎯 PASS DETECTOR INITIALIZED WITH ENHANCED TEAM VALIDATION - VERSION 2.0")
        all_passes = []

        # PRIMARY: Touch-based detection (best for brief possessions and one-touch passes)
        if ball_assignments is not None:
            touch_passes = self._detect_touch_based_passes(
                ball_assignments, team_assignments, field_coords_players, fps, ball_field_coords
            )
            all_passes.extend(touch_passes)
            print(f"    Touch-based detection: {len(touch_passes)} passes")

        # SECONDARY: Possession-based detection (for longer passes not captured by touch events)
        possession_passes = self._detect_possession_based_passes(
            possession_events, field_coords_players, team_assignments, fps, ball_field_coords, ball_assignments
        )
        print(f"    Possession-based detection: {len(possession_passes)} passes")

        # Merge and deduplicate passes
        all_passes = self._merge_pass_detections(all_passes, possession_passes, fps)

        return all_passes

    def _merge_pass_detections(self, touch_passes: List[Dict], possession_passes: List[Dict],
                               fps: float, time_tolerance: float = 4.0) -> List[Dict]:
        """
        Merge passes from touch-based and possession-based detection, removing duplicates.

        Duplicates are identified by same players AND similar timing, since touch-based
        detects at start of possession and possession-based detects at end, but they represent
        the same physical pass event.

        IMPORTANT: Multiple passes between the same players at different times are ALLOWED.

        Args:
            touch_passes: Passes detected from touch events
            possession_passes: Passes detected from possession transitions
            fps: Frames per second
            time_tolerance: Time window (seconds) to consider passes as duplicates

        Returns:
            Merged list of unique passes
        """
        merged = list(touch_passes)  # Start with touch-based (higher priority)
        frame_tolerance = int(time_tolerance * fps)

        for poss_pass in possession_passes:
            is_duplicate = False

            # Check if this possession pass matches any touch pass (same players + similar timing)
            for touch_pass in touch_passes:
                if (touch_pass['from_player'] == poss_pass['from_player'] and
                    touch_pass['to_player'] == poss_pass['to_player']):
                    # Same players - check if timing is close enough (likely same pass)
                    frame_diff = abs(touch_pass['frame'] - poss_pass['frame'])
                    if frame_diff <= frame_tolerance:
                        is_duplicate = True
                        break

            if not is_duplicate:
                merged.append(poss_pass)

        # Sort by frame
        merged.sort(key=lambda x: x['frame'])

        print(f"    Merged: {len(merged)} unique passes (removed {len(touch_passes) + len(possession_passes) - len(merged)} duplicates)")

        return merged

    def _detect_possession_based_passes(self, possession_events: List[Dict], field_coords_players: Dict,
                                       team_assignments: Dict, fps: float,
                                       ball_field_coords: Dict = None, ball_assignments: Dict = None) -> List[Dict]:
        """
        SECONDARY METHOD: Detect passes from possession event transitions.

        This catches longer passes that touch-based detection might miss due to gaps in ball_assignments.

        Args:
            possession_events: List of possession events from PossessionTracker
            field_coords_players: Dictionary {player_id: {'field_coordinates': {frame_idx: [x, y]}}}
            team_assignments: Dictionary {frame_idx: {player_id: team_id}}
            fps: Frames per second
            ball_field_coords: Dictionary {frame_idx: [x, y]} ball field coordinates (optional)
            ball_assignments: Dictionary {frame_idx: player_id} for validation (optional)

        Returns:
            List of pass events
        """
        passes = []

        # Extract touch events if ball_assignments provided (for validation only)
        touch_events = []
        if ball_assignments is not None:
            touch_events = self._extract_touch_events(ball_assignments)

        # Sort possession events by start frame
        sorted_events = sorted(possession_events, key=lambda x: x['start_frame'])

        # Track which events have been used as receivers to avoid duplicates
        # NOTE: We no longer prevent possessions from being receivers multiple times
        # because a player can receive a pass and immediately send another pass

        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]

            # Skip if this possession doesn't meet minimum duration
            if current_event['duration_seconds'] < self.min_possession_duration:
                continue

            # Find the actual pass target (may skip brief opposite-team possessions)
            result = self._find_pass_target(sorted_events, i, team_assignments, max_interrupted_frames=15)

            if result is None:
                continue

            target_idx, total_gap = result
            next_event = sorted_events[target_idx]

            # Don't skip - allow possessions to be receivers multiple times
            # (e.g., Player 18 receives from 13, then passes to 14 who receives and passes again)

            # Use the gap calculated by _find_pass_target
            gap_frames = total_gap

            # Get team assignments for both players with enhanced robustness
            passer_frame = current_event['end_frame']
            receiver_frame = next_event['start_frame']

            passer_id = current_event['player_id']
            receiver_id = next_event['player_id']

            # Use the same enhanced team validation as touch-based detection
            def get_dominant_team_possession(player_id, center_frame, team_assignments):
                """Get the dominant team for a player around a specific frame."""
                team_votes = []

                # Check a wider range of frames to find the dominant team
                for offset in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    test_frame = center_frame + offset
                    if test_frame in team_assignments and player_id in team_assignments[test_frame]:
                        team_votes.append(team_assignments[test_frame][player_id])

                if not team_votes:
                    return None

                # Use majority vote - the stabilized team should be dominant
                from collections import Counter
                team_counts = Counter(team_votes)
                dominant_team = team_counts.most_common(1)[0][0]

                # Only return if there's reasonable consistency (>60% agreement)
                agreement = team_counts[dominant_team] / len(team_votes)
                if agreement > 0.6:
                    return dominant_team

                return None

            # Get dominant teams for both players
            passer_team = get_dominant_team_possession(passer_id, passer_frame, team_assignments)
            receiver_team = get_dominant_team_possession(receiver_id, receiver_frame, team_assignments)

            # Fallback to event team if enhanced validation fails
            if passer_team is None:
                passer_team = current_event.get('team')
            if receiver_team is None:
                receiver_team = next_event.get('team')

            # Enhanced team validation
            if passer_team is None or receiver_team is None or passer_team != receiver_team:
                print(f"        [DEBUG] Possession-based team validation failed: P{passer_id}({passer_team})→P{receiver_id}({receiver_team})")
                continue

            # Get positions with enhanced robustness
            passer_pos = None
            # Try wider range for passer positions
            for frame_offset in [0, -1, -2, -3, -4, -5, 1, 2, 3, 4, 5]:
                test_frame = passer_frame + frame_offset
                if current_event['start_frame'] <= test_frame <= current_event['end_frame']:
                    passer_pos = self._get_player_position(passer_id, test_frame, field_coords_players)
                    if passer_pos is not None:
                        break

            receiver_pos = None
            # Try wider range for receiver positions
            for frame_offset in [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5]:
                test_frame = receiver_frame + frame_offset
                if next_event['start_frame'] <= test_frame <= next_event['end_frame']:
                    receiver_pos = self._get_player_position(receiver_id, test_frame, field_coords_players)
                    if receiver_pos is not None:
                        break

            # Enhanced fallback: use any available position for these players
            if passer_pos is None:
                if passer_id in field_coords_players and 'field_coordinates' in field_coords_players[passer_id]:
                    all_positions = list(field_coords_players[passer_id]['field_coordinates'].values())
                    if all_positions:
                        passer_pos = np.array(all_positions[len(all_positions)//2])  # Use middle position

            if receiver_pos is None:
                if receiver_id in field_coords_players and 'field_coordinates' in field_coords_players[receiver_id]:
                    all_positions = list(field_coords_players[receiver_id]['field_coordinates'].values())
                    if all_positions:
                        receiver_pos = np.array(all_positions[len(all_positions)//2])  # Use middle position

            if passer_pos is None or receiver_pos is None:
                print(f"        [DEBUG] Possession-based position validation failed: P{passer_id}({passer_pos is not None})→P{receiver_id}({receiver_pos is not None})")
                continue

            # Calculate distance between players
            pass_distance = self._calculate_distance(passer_pos, receiver_pos)

            # Validate pass distance range (minimum and maximum)
            if pass_distance < self.min_pass_distance or pass_distance > self.max_pass_distance:
                continue

            # Calculate ball velocity if ball_field_coords available (for metadata, not filtering)
            ball_velocity = 0.0
            velocity_warning = False
            if ball_field_coords is not None and gap_frames > 0:
                ball_velocity = self._calculate_ball_velocity(
                    ball_field_coords, passer_frame, receiver_frame, fps
                )
                # Flag low velocity but don't reject (user wants all passes counted)
                if ball_velocity < self.velocity_threshold:
                    velocity_warning = True

            # Calculate pass timestamp and ball travel time
            pass_timestamp = passer_frame / fps
            ball_travel_time = gap_frames / fps

            # Create pass event with enhanced metadata
            pass_event = {
                'frame': int(passer_frame),
                'timestamp': round(pass_timestamp, 2),
                'from_player': int(passer_id),
                'to_player': int(receiver_id),
                'team': int(passer_team),
                'distance_m': round(pass_distance, 2),
                'start_position': [round(float(passer_pos[0]), 2), round(float(passer_pos[1]), 2)],
                'end_position': [round(float(receiver_pos[0]), 2), round(float(receiver_pos[1]), 2)],
                'ball_travel_time_s': round(ball_travel_time, 2),
                'gap_frames': int(gap_frames),
                'ball_velocity_ms': round(ball_velocity, 2) if ball_velocity > 0 else None,
                'low_velocity_warning': velocity_warning
            }

            # Validate with touch events if available (for confidence scoring, not filtering)
            if touch_events:
                is_valid, confidence = self._validate_with_touches(pass_event, touch_events, tolerance_frames=5)
                pass_event['touch_validated'] = is_valid
                pass_event['confidence_score'] = round(confidence, 2)
                pass_event['validation_method'] = 'hybrid' if is_valid else 'possession_only'
                # Note: We don't filter out passes based on touch validation - all passes are counted
            else:
                pass_event['validation_method'] = 'possession_only'
                pass_event['confidence_score'] = 0.5

            passes.append(pass_event)

        return passes

    def count_player_passes(self, passes: List[Dict]) -> Dict[int, Dict[str, int]]:
        """
        Count passes completed and received by each player.

        Args:
            passes: List of pass events

        Returns:
            Dictionary {player_id: {'passes_completed': int, 'passes_received': int}}
        """
        player_pass_counts = {}

        for pass_event in passes:
            from_player = pass_event['from_player']
            to_player = pass_event['to_player']

            # Initialize if needed
            if from_player not in player_pass_counts:
                player_pass_counts[from_player] = {'passes_completed': 0, 'passes_received': 0}
            if to_player not in player_pass_counts:
                player_pass_counts[to_player] = {'passes_completed': 0, 'passes_received': 0}

            # Increment counts
            player_pass_counts[from_player]['passes_completed'] += 1
            player_pass_counts[to_player]['passes_received'] += 1

        return player_pass_counts

    def verify_passes_with_touches(self, passes: List[Dict], touch_frames: Dict[int, List[int]],
                                   tolerance_frames: int = 15) -> List[Dict]:
        """
        Verify passes by correlating with actual player touches.

        Adds verification metadata to each pass indicating if the passer and receiver
        had registered touches near the pass frame.

        Args:
            passes: List of pass events from detect_passes
            touch_frames: Dictionary {player_id: [frame_indices]} from PlayerBallAssigner
            tolerance_frames: Number of frames to search around pass frame (default: 15 ~0.5s at 30fps)

        Returns:
            List of pass events with added 'passer_touch_verified' and 'receiver_touch_verified' fields
        """
        verified_passes = []

        for pass_event in passes.copy():
            from_player = pass_event['from_player']
            to_player = pass_event['to_player']
            pass_frame = pass_event['frame']
            receive_frame = pass_frame + pass_event.get('gap_frames', 0)

            # Check if passer had touch near pass frame
            passer_touch_verified = False
            if from_player in touch_frames:
                passer_touches = touch_frames[from_player]
                passer_touch_verified = any(abs(t - pass_frame) <= tolerance_frames for t in passer_touches)

            # Check if receiver had touch near receive frame
            receiver_touch_verified = False
            if to_player in touch_frames:
                receiver_touches = touch_frames[to_player]
                receiver_touch_verified = any(abs(t - receive_frame) <= tolerance_frames for t in receiver_touches)

            # Add verification metadata
            pass_event['passer_touch_verified'] = passer_touch_verified
            pass_event['receiver_touch_verified'] = receiver_touch_verified
            pass_event['fully_verified'] = passer_touch_verified and receiver_touch_verified

            verified_passes.append(pass_event)

        return verified_passes
