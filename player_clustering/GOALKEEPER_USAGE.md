# Goalkeeper Assignment Usage

## Overview

The `GoalkeeperAssigner` module assigns goalkeepers to teams based on spatial proximity to team centroids. This is useful when goalkeepers wear different colors than their field players.

## Quick Start

```python
from player_clustering import GoalkeeperAssigner

# Initialize assigner
gk_assigner = GoalkeeperAssigner()

# Assign goalkeepers to teams (using supervision Detections)
goalkeeper_team_ids = gk_assigner.assign_goalkeeper_teams(
    players=player_detections,        # Field player detections
    goalkeepers=goalkeeper_detections, # Goalkeeper detections
    player_team_ids=player_team_ids   # Team IDs for field players (0 or 1)
)
```

## Example with Tracking Data

```python
from player_clustering import GoalkeeperAssigner

gk_assigner = GoalkeeperAssigner()

# Assign goalkeepers for a specific frame
frame_idx = 100
gk_assignments = gk_assigner.assign_goalkeeper_teams_from_tracks(
    player_tracks=player_tracks,
    goalkeeper_tracks=goalkeeper_tracks,
    team_assignments=team_assignments,
    frame_idx=frame_idx
)

# Result: {gk_id: team_id}
print(f"Goalkeeper assignments: {gk_assignments}")
```

## How It Works

1. **Calculate Team Centroids**: Computes the average position of each team's field players
2. **Measure Distances**: Calculates Euclidean distance from each goalkeeper to both team centroids
3. **Assign to Closest Team**: Assigns each goalkeeper to the team with the closest centroid

## Integration with Existing Pipeline

The goalkeeper assigner can be integrated into your tracking pipeline:

```python
# After getting team assignments for field players
from player_clustering import ClusteringManager, GoalkeeperAssigner

clustering_manager = ClusteringManager()
gk_assigner = GoalkeeperAssigner()

# Train on field players
clustering_manager.fit(field_player_crops)

# Get team assignments for field players
field_player_teams = clustering_manager.predict(field_player_crops)

# Assign goalkeepers based on field player positions
goalkeeper_teams = gk_assigner.assign_goalkeeper_teams(
    players=field_player_detections,
    goalkeepers=goalkeeper_detections,
    player_team_ids=field_player_teams
)
```

## Edge Cases Handled

- **No goalkeepers detected**: Returns empty array
- **No field players**: Defaults all goalkeepers to team 0
- **One team missing**: Assigns all goalkeepers to the existing team
- **Multiple goalkeepers**: Each is independently assigned to closest team

## Benefits

- ✅ Works when goalkeepers wear different colors
- ✅ No need to train separate models
- ✅ Fast computation (simple distance calculation)
- ✅ Robust to camera angles and zoom levels
