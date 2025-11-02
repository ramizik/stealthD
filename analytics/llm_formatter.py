"""
LLM Data Formatter - Creates compact, LLM-ready JSON from full analysis data.

Converts frame-by-frame tracking data into summarized metrics optimized
for LLM consumption and coaching insights.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from typing import Dict, List, Any
import time


class LLMDataFormatter:
    """
    Format full analysis data into compact LLM-ready JSON.

    Removes frame-by-frame data and provides only aggregated metrics
    suitable for LLM-based coaching analysis.
    """

    def __init__(self):
        """Initialize the LLM data formatter."""
        pass

    def format_for_llm(self, full_analysis_data: Dict, video_name: str) -> Dict:
        """
        Convert full analysis data to LLM-ready compact format.

        Args:
            full_analysis_data: Full JSON from _prepare_analysis_json()
            video_name: Name of the video file

        Returns:
            Compact dictionary optimized for LLM consumption
        """
        return {
            "match_summary": self._create_match_summary(full_analysis_data, video_name),
            "match_context": self._create_match_context(full_analysis_data),
            "player_statistics": self._summarize_players(full_analysis_data),
            "team_statistics": self._summarize_teams(full_analysis_data),
            "match_events_summary": self._summarize_events(full_analysis_data),
            "key_insights": self._generate_insights(full_analysis_data)
        }

    def _create_match_summary(self, data: Dict, video_name: str) -> Dict:
        """
        Create high-level match summary.

        Args:
            data: Full analysis data
            video_name: Video filename

        Returns:
            Match summary dictionary
        """
        metadata = data.get('metadata', {})

        # Calculate duration in seconds
        frames = metadata.get('frames_processed', 0)
        fps = metadata.get('fps', 30)
        duration_seconds = frames / fps if fps > 0 else 0

        return {
            "video_name": Path(video_name).stem,
            "date": metadata.get('timestamp', time.strftime("%Y-%m-%d")),
            "duration_seconds": round(duration_seconds, 1),
            "total_frames": frames,
            "fps": fps,
            "processing_time_seconds": metadata.get('processing_time_seconds', 0)
        }

    def _create_match_context(self, data: Dict) -> Dict:
        """
        Create match context information.

        Args:
            data: Full analysis data

        Returns:
            Match context dictionary with field dimensions and teams
        """
        metadata = data.get('metadata', {})
        field_dims = metadata.get('field_dimensions', {'width': 105, 'height': 68})

        # Get team assignments from player analytics
        player_analytics = data.get('player_analytics', {})
        team_0_players = []
        team_1_players = []

        for player_id, analytics in player_analytics.items():
            team = analytics.get('team', None)
            player_id_int = int(player_id)
            if team == 0:
                team_0_players.append(player_id_int)
            elif team == 1:
                team_1_players.append(player_id_int)

        return {
            "field_dimensions_m": {
                "length": field_dims.get('width', 105),
                "width": field_dims.get('height', 68)
            },
            "teams": {
                "team_0": {
                    "name": "Team 0",
                    "color": "Blue",
                    "player_ids": sorted(team_0_players)
                },
                "team_1": {
                    "name": "Team 1",
                    "color": "Red",
                    "player_ids": sorted(team_1_players)
                }
            }
        }

    def _summarize_players(self, data: Dict) -> Dict:
        """
        Summarize individual player statistics.

        Args:
            data: Full analysis data

        Returns:
            Dictionary of player statistics
        """
        player_analytics = data.get('player_analytics', {})
        player_speeds = data.get('player_speeds', {})
        passes = data.get('passes', [])

        player_summaries = {}

        for player_id, analytics in player_analytics.items():
            player_id_int = int(player_id)

            # Basic info
            team = analytics.get('team', None)
            position = self._estimate_position(player_id_int, team, player_speeds)

            # Performance metrics
            performance = {
                "distance_covered_km": round(analytics.get('total_distance_m', 0) / 1000, 2),
                "average_speed_kmh": analytics.get('average_speed_kmh', 0),
                "max_speed_kmh": analytics.get('max_speed_kmh', 0),
                "sprint_count": self._count_sprints(player_id_int, player_speeds)
            }

            # Possession metrics
            possession_seconds = analytics.get('ball_possession_seconds', 0)
            # Use frame-by-frame touch count if available, otherwise fall back to old method
            touch_count = analytics.get('ball_touches', self._count_touches(player_id_int, data))
            possession = {
                "possession_time_seconds": round(possession_seconds, 1),
                "possession_percentage": round(possession_seconds / (data.get('metadata', {}).get('frames_processed', 1) / data.get('metadata', {}).get('fps', 30)) * 100, 2) if possession_seconds > 0 else 0,
                "touches": touch_count
            }

            # Passing metrics
            passes_completed = analytics.get('passes_completed', 0)
            passes_received = analytics.get('passes_received', 0)
            avg_pass_distance = self._calculate_avg_pass_distance(player_id_int, passes)

            passing = {
                "passes_completed": passes_completed,
                "passes_received": passes_received,
                "pass_accuracy_percentage": self._calculate_pass_accuracy(player_id_int, passes),
                "average_pass_distance_m": round(avg_pass_distance, 1) if avg_pass_distance else 0
            }

            # Shooting metrics (placeholder for future implementation)
            shooting = {
                "shots_total": 0,
                "shots_on_target": 0,
                "shot_conversion_rate": 0.0
            }

            player_summaries[str(player_id_int)] = {
                "name": f"Player #{player_id_int}",
                "team": team,
                "position_estimate": position,
                "performance": performance,
                "possession": possession,
                "passing": passing,
                "shooting": shooting
            }

        return player_summaries

    def _summarize_teams(self, data: Dict) -> Dict:
        """
        Summarize team-level statistics.

        Args:
            data: Full analysis data

        Returns:
            Dictionary of team statistics for both teams
        """
        player_analytics = data.get('player_analytics', {})
        passes = data.get('passes', [])
        possession_data = data.get('ball_possession', {})
        metadata = data.get('metadata', {})

        # Separate players by team
        team_0_players = []
        team_1_players = []

        for player_id, analytics in player_analytics.items():
            team = analytics.get('team', None)
            if team == 0:
                team_0_players.append((int(player_id), analytics))
            elif team == 1:
                team_1_players.append((int(player_id), analytics))

        # Calculate team statistics
        team_stats = {}
        for team_id, team_name, players in [(0, "Team 0", team_0_players), (1, "Team 1", team_1_players)]:
            team_stats[f"team_{team_id}"] = self._calculate_team_metrics(
                team_id, team_name, players, passes, possession_data, metadata
            )

        return team_stats

    def _calculate_team_metrics(self, team_id: int, team_name: str,
                                players: List[tuple], passes: List[Dict],
                                possession_data: Dict, metadata: Dict) -> Dict:
        """
        Calculate comprehensive team metrics.

        Args:
            team_id: Team identifier (0 or 1)
            team_name: Team name
            players: List of (player_id, analytics) tuples
            passes: List of pass events
            possession_data: Possession tracking data
            metadata: Match metadata

        Returns:
            Dictionary of team metrics
        """
        if not players:
            return self._empty_team_metrics(team_name)

        # Overall performance
        total_distance_km = sum(p[1].get('total_distance_m', 0) for p in players) / 1000
        avg_distance_km = total_distance_km / len(players) if players else 0
        avg_speed = sum(p[1].get('average_speed_kmh', 0) for p in players) / len(players) if players else 0

        # Possession metrics
        total_possession = sum(p[1].get('ball_possession_seconds', 0) for p in players)
        possession_events = possession_data.get('possession_events', [])
        team_possession_events = [e for e in possession_events if self._get_player_team(e.get('player_id'), players) == team_id]

        # Passing metrics
        team_passes = [p for p in passes if p.get('team') == team_id]
        passes_completed = len(team_passes)

        # Calculate turnovers (passes to other team or possession lost)
        turnovers = self._count_turnovers(team_id, possession_events, players)

        return {
            "name": team_name,
            "overall": {
                "total_distance_km": round(total_distance_km, 2),
                "average_distance_per_player_km": round(avg_distance_km, 2),
                "average_speed_kmh": round(avg_speed, 1)
            },
            "possession": {
                "possession_percentage": round(total_possession / (metadata.get('frames_processed', 1) / metadata.get('fps', 30)) * 100, 1),
                "total_possession_seconds": round(total_possession, 1),
                "possession_events": len(team_possession_events),
                "average_possession_duration_seconds": round(total_possession / len(team_possession_events), 1) if team_possession_events else 0
            },
            "passing": {
                "passes_completed": passes_completed,
                "passes_attempted": passes_completed,  # Will be updated when we track failed passes
                "pass_completion_percentage": 100.0,  # Placeholder
                "passes_per_possession": round(passes_completed / len(team_possession_events), 1) if team_possession_events else 0,
                "turnovers": turnovers,
                "interceptions_suffered": 0  # Placeholder
            },
            "shooting": {
                "shots_total": 0,
                "shots_on_target": 0,
                "shot_accuracy_percentage": 0.0,
                "goals": 0
            },
            "formation": {
                "detected_formation": "Unknown",
                "formation_confidence": 0.0,
                "formation_changes": 0
            },
            "spacing": {
                "average_width_m": 0.0,
                "average_compactness_m": 0.0,
                "defensive_line_depth_m": 0.0
            },
            "pressing": {
                "successful_presses": 0,
                "press_success_rate": 0.0,
                "average_press_duration_seconds": 0.0
            },
            "ball_recoveries": {
                "defensive_third": 0,
                "midfield": 0,
                "attacking_third": 0,
                "total": 0,
                "average_recovery_time_seconds": 0.0
            },
            "transitions": {
                "counter_attacks": 0,
                "counter_attack_success_rate": 0.0
            }
        }

    def _summarize_events(self, data: Dict) -> Dict:
        """
        Summarize key match events.

        Args:
            data: Full analysis data

        Returns:
            Dictionary of event summaries
        """
        passes = data.get('passes', [])
        possession_events = data.get('ball_possession', {}).get('possession_events', [])
        player_analytics = data.get('player_analytics', {})

        # Find most active players
        most_active = max(player_analytics.items(),
                         key=lambda x: x[1].get('ball_possession_seconds', 0))[0] if player_analytics else None
        top_passer = max(player_analytics.items(),
                        key=lambda x: x[1].get('passes_completed', 0))[0] if player_analytics else None

        # Calculate possession changes
        possession_changes = len(possession_events) - 1 if len(possession_events) > 1 else 0

        # Calculate average and longest possession
        possession_durations = [e.get('duration_seconds', 0) for e in possession_events]
        avg_possession = sum(possession_durations) / len(possession_durations) if possession_durations else 0
        longest_possession = max(possession_durations) if possession_durations else 0

        return {
            "total_passes": len(passes),
            "total_shots": 0,  # Placeholder
            "total_possession_changes": possession_changes,
            "average_possession_duration_seconds": round(avg_possession, 1),
            "longest_possession_seconds": round(longest_possession, 1),
            "most_active_player": int(most_active) if most_active else None,
            "top_passer": int(top_passer) if top_passer else None,
            "top_shooter": None  # Placeholder
        }

    def _generate_insights(self, data: Dict) -> Dict:
        """
        Generate high-level insights from the data.

        Args:
            data: Full analysis data

        Returns:
            Dictionary of key insights
        """
        player_analytics = data.get('player_analytics', {})

        # Separate teams
        team_0_possession = sum(p.get('ball_possession_seconds', 0)
                               for p in player_analytics.values()
                               if p.get('team') == 0)
        team_1_possession = sum(p.get('ball_possession_seconds', 0)
                               for p in player_analytics.values()
                               if p.get('team') == 1)

        team_0_passes = sum(p.get('passes_completed', 0)
                           for p in player_analytics.values()
                           if p.get('team') == 0)
        team_1_passes = sum(p.get('passes_completed', 0)
                           for p in player_analytics.values()
                           if p.get('team') == 1)

        # Determine dominance
        possession_dominance = "team_0" if team_0_possession > team_1_possession else "team_1"
        passing_leader = "team_0" if team_0_passes > team_1_passes else "team_1"

        return {
            "possession_dominance": possession_dominance,
            "passing_efficiency_leader": passing_leader,
            "most_aggressive_team": "unknown",  # Placeholder
            "formation_stability": "medium",  # Placeholder
            "tempo": "medium"  # Placeholder
        }

    # Helper methods

    def _estimate_position(self, player_id: int, team: int, player_speeds: Dict) -> str:
        """Estimate player position based on average field position."""
        # Placeholder - will be improved with formation detection
        if player_id in [1, 2, 3, 4, 12, 13, 14, 15]:
            return "Defender"
        elif player_id in [5, 6, 7, 8, 16, 17, 18, 19]:
            return "Midfielder"
        else:
            return "Forward"

    def _count_sprints(self, player_id: int, player_speeds: Dict) -> int:
        """Count number of sprints (speed > 20 km/h)."""
        if str(player_id) not in player_speeds:
            return 0

        frames_data = player_speeds[str(player_id)].get('frames', {})
        sprint_threshold = 20.0  # km/h

        sprints = sum(1 for speed in frames_data.values() if speed > sprint_threshold)
        return sprints

    def _count_touches(self, player_id: int, data: Dict) -> int:
        """Estimate number of ball touches based on possession events."""
        possession_events = data.get('ball_possession', {}).get('possession_events', [])
        touches = sum(1 for e in possession_events if e.get('player_id') == player_id)
        return touches

    def _calculate_avg_pass_distance(self, player_id: int, passes: List[Dict]) -> float:
        """Calculate average pass distance for a player."""
        player_passes = [p for p in passes if p.get('from_player') == player_id]
        if not player_passes:
            return 0.0

        distances = [p.get('distance_m', 0) for p in player_passes]
        return sum(distances) / len(distances) if distances else 0.0

    def _calculate_pass_accuracy(self, player_id: int, passes: List[Dict]) -> float:
        """Calculate pass accuracy percentage for a player."""
        # For now, all recorded passes are successful
        # Will be updated when we track failed passes
        player_passes = [p for p in passes if p.get('from_player') == player_id]
        return 100.0 if player_passes else 0.0

    def _get_player_team(self, player_id: int, players: List[tuple]) -> int:
        """Get team ID for a player."""
        for pid, analytics in players:
            if pid == player_id:
                return analytics.get('team', -1)
        return -1

    def _count_turnovers(self, team_id: int, possession_events: List[Dict],
                        players: List[tuple]) -> int:
        """Count turnovers (possession lost to opponent)."""
        turnovers = 0
        prev_team = None

        for event in possession_events:
            player_id = event.get('player_id')
            current_team = self._get_player_team(player_id, players)

            if prev_team is not None and prev_team == team_id and current_team != team_id:
                turnovers += 1

            prev_team = current_team

        return turnovers

    def _empty_team_metrics(self, team_name: str) -> Dict:
        """Return empty team metrics structure."""
        return {
            "name": team_name,
            "overall": {"total_distance_km": 0, "average_distance_per_player_km": 0, "average_speed_kmh": 0},
            "possession": {"possession_percentage": 0, "total_possession_seconds": 0, "possession_events": 0, "average_possession_duration_seconds": 0},
            "passing": {"passes_completed": 0, "passes_attempted": 0, "pass_completion_percentage": 0, "passes_per_possession": 0, "turnovers": 0, "interceptions_suffered": 0},
            "shooting": {"shots_total": 0, "shots_on_target": 0, "shot_accuracy_percentage": 0, "goals": 0},
            "formation": {"detected_formation": "Unknown", "formation_confidence": 0.0, "formation_changes": 0},
            "spacing": {"average_width_m": 0.0, "average_compactness_m": 0.0, "defensive_line_depth_m": 0.0},
            "pressing": {"successful_presses": 0, "press_success_rate": 0.0, "average_press_duration_seconds": 0.0},
            "ball_recoveries": {"defensive_third": 0, "midfield": 0, "attacking_third": 0, "total": 0, "average_recovery_time_seconds": 0.0},
            "transitions": {"counter_attacks": 0, "counter_attack_success_rate": 0.0}
        }
