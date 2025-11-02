# Soccer Analysis MVP Development Plan

## Project Overview

This document outlines the development plan for the Soccer Analysis MVP, focusing on data collection and analytics required for LLM-powered coaching feedback. The goal is to provide comprehensive individual player feedback and team-level tactical suggestions.

---

## Current Project Status

### ✅ **JSON Output System**

The project now generates **two JSON files** per analysis:

1. **Full Analysis JSON** (`{video}_analysis_data.json`)
   - Contains all frame-by-frame tracking data
   - Size: ~185K lines
   - Used for debugging and detailed replay analysis
   - Includes: `player_tracks`, `ball_tracks`, `referee_tracks`, `team_assignments` per frame

2. **LLM-Ready Compact JSON** (`{video}_analysis_data_llm.json`) ✨ **NEW**
   - Contains only aggregated/summarized metrics
   - Size: ~300-500 lines (~95% size reduction)
   - Optimized for LLM context windows
   - Perfect for coaching insights and natural language queries
   - **Implementation:** `analytics/llm_formatter.py`

### ✅ **Implemented Individual Player Metrics**

The following player-level analytics are currently being tracked and exported to JSON:

#### Available Data Structure:
```json
{
  "player_analytics": {
    "player_id": {
      "team": 0 or 1,
      "average_speed_kmh": float,
      "max_speed_kmh": float,
      "total_distance_m": float,
      "ball_possession_seconds": float,
      "passes_completed": int,
      "passes_received": int
    }
  },
  "player_speeds": {
    "player_id": {
      "frames": {frame_idx: speed_kmh},
      "average_speed_kmh": float,
      "max_speed_kmh": float,
      "total_distance_m": float,
      "field_coordinates": {frame_idx: [x, y]}
    }
  },
  "ball_possession": {
    "possession_events": [...],
    "player_possession_time": {player_id: seconds}
  },
  "passes": [
    {
      "from_player": int,
      "to_player": int,
      "distance_m": float,
      "start_frame": int,
      "end_frame": int,
      "team": int
    }
  ]
}
```

#### Individual Player Metrics Status:
- ✅ **Ball holding time** - `ball_possession_seconds` per player ✓
- ✅ **Pass counts** - `passes_completed` + `passes_received` ✓
- ✅ **Distance covered** - `total_distance_m` ✓
- ✅ **Average speed** - `average_speed_kmh` + `max_speed_kmh` ✓
- ❌ **Shots on goal** - **NOT IMPLEMENTED**

---

### ❌ **Missing Team-Level Tactical Metrics**

Currently, **NO team-level aggregation** exists. All data is per-player only. The following team tactical metrics are required for MVP:

#### 1. **Possession & Control**
- ❌ **Possession %** - Team possession time vs total match time
- ❌ **Pass completion %** - Successful passes / total pass attempts per team
- ❌ **Turnovers** - Count of possession switches to opposing team

#### 2. **Formation & Spacing**
- ❌ **Width** - Average horizontal spread of team players (meters)
- ❌ **Compactness** - Average distance between team players (density metric)
- ❌ **Formation confidence** - Detection of formation pattern (4-4-2, 4-3-3, etc.)

#### 3. **Transitions**
- ❌ **Counter-attack frequency** - Rapid possession changes + forward movement detection
- ❌ **Press recovery time** - Time from losing possession to regaining it

#### 4. **Pressing**
- ❌ **Press intensity index** - Distance covered in defensive third / time spent pressing
- ❌ **Number of successful presses** - Possession regained within X seconds in opponent's half
- ❌ **Ball recoveries** - Count of possession regains per team/zone (defensive/midfield/attacking third)

#### 5. **Defensive Shape**
- ❌ **Gaps between lines** - Distance between defensive/midfield/attacking lines
- ❌ **Off-ball movement synchronization** - Correlation of player movements without ball

---

## MVP Implementation Plan

### **Phase 1: Essential Team Aggregations** (Priority: CRITICAL)

**Goal:** Provide basic team-level metrics for LLM to answer fundamental coaching questions.

#### Task 1.1: Team Possession Calculator
**File:** `analytics/team_possession_calculator.py`

**Requirements:**
- Aggregate `ball_possession_seconds` by team
- Calculate possession percentage: `team_possession_time / total_match_time * 100`
- Output per-team metrics:
  ```json
  {
    "team_0": {
      "total_possession_seconds": 450.5,
      "possession_percentage": 52.3,
      "possession_events_count": 45
    },
    "team_1": {...}
  }
  ```

**Dependencies:**
- Existing `possession_data` from `PossessionTracker`
- Team assignments from `all_tracks['player_classids']`

---

#### Task 1.2: Pass Completion Rate Calculator
**File:** `analytics/pass_completion_calculator.py`

**Requirements:**
- Detect failed passes (ball lost between same-team players, or intercepted by opponent)
- Calculate pass completion rate: `successful_passes / (successful_passes + failed_passes) * 100`
- Track turnovers (possession switches from Team A → Team B)
- Output per-team metrics:
  ```json
  {
    "team_0": {
      "passes_completed": 156,
      "passes_failed": 42,
      "pass_completion_percentage": 78.8,
      "turnovers": 18,
      "interceptions": 12
    },
    "team_1": {...}
  }
  ```

**Dependencies:**
- Existing `passes` list from `PassDetector`
- Possession events to detect failed passes
- Team assignments

---

#### Task 1.3: Shots Detection Module
**File:** `analytics/shot_detector.py`

**Requirements:**
- Detect shots: Ball moving rapidly toward goal (target area) from player position
- Define goal zones: Left goal (x: 0-5m, y: 25-43m), Right goal (x: 100-105m, y: 25-43m)
- Validate shots: Ball velocity > threshold, direction toward goal, player proximity
- Track shot outcomes (if possible): Goal, save, miss, blocked
- Output per-player and per-team:
  ```json
  {
    "player_shots": {
      "5": {
        "shots_on_goal": 3,
        "shots_total": 5,
        "shot_locations": [[x, y], ...]
      }
    },
    "team_shots": {
      "team_0": {
        "shots_total": 12,
        "shots_on_target": 8,
        "shots_off_target": 4
      }
    }
  }
  ```

**Dependencies:**
- Ball tracks
- Player positions
- Field coordinates
- Goal area definitions

---

#### Task 1.4: Team Distance Aggregator
**File:** `analytics/team_distance_calculator.py`

**Requirements:**
- Sum total distance covered by all players per team
- Calculate average distance per player per team
- Output:
  ```json
  {
    "team_0": {
      "total_distance_km": 45.2,
      "average_distance_per_player_km": 4.1,
      "players": {
        "1": 4.5,
        "2": 3.8,
        ...
      }
    }
  }
  ```

**Dependencies:**
- Existing `player_speeds` with `total_distance_m`

---

#### Task 1.5: Team Analytics Orchestrator
**File:** `analytics/team_analytics_calculator.py`

**Requirements:**
- Coordinate all team-level calculations
- Aggregate results into unified structure
- Integrate with existing `MetricsCalculator`

**Output Structure:**
```json
{
  "team_analytics": {
    "team_0": {
      "possession": {
        "possession_percentage": 52.3,
        "total_possession_seconds": 450.5
      },
      "passing": {
        "pass_completion_percentage": 78.8,
        "passes_completed": 156,
        "passes_failed": 42,
        "turnovers": 18
      },
      "shooting": {
        "shots_total": 12,
        "shots_on_target": 8
      },
      "distance": {
        "total_distance_km": 45.2,
        "average_per_player_km": 4.1
      }
    },
    "team_1": {...}
  }
}
```

---

### **Phase 2: Tactical Metrics** (Priority: HIGH)

**Goal:** Enable LLM to provide tactical coaching suggestions.

#### Task 2.1: Formation Detector
**File:** `analytics/formation_detector.py`

**Requirements:**
- Analyze player positions to identify formation pattern
- Common formations: 4-4-2, 4-3-3, 3-5-2, 4-2-3-1
- Use clustering to identify defensive/midfield/forward lines
- Calculate formation confidence score
- Output:
  ```json
  {
    "team_0": {
      "detected_formation": "4-3-3",
      "formation_confidence": 0.85,
      "formation_alternatives": {
        "4-4-2": 0.45,
        "4-3-3": 0.85
      },
      "player_positions_by_line": {
        "defenders": [1, 2, 3, 4],
        "midfielders": [5, 6, 7],
        "forwards": [8, 9, 10]
      }
    }
  }
  ```

**Algorithm Approach:**
1. Divide field into defensive/midfield/attacking zones
2. Cluster players by average Y position (field length)
3. Count players per zone
4. Match to known formation patterns
5. Calculate confidence based on position consistency

---

#### Task 2.2: Team Spacing Calculator
**File:** `analytics/spacing_calculator.py`

**Requirements:**
- **Width:** Maximum horizontal spread (max X - min X) of all team players
- **Compactness:** Average distance between all team player pairs
- Calculate per-frame, then average across match
- Output:
  ```json
  {
    "team_0": {
      "average_width_m": 42.5,
      "average_compactness_m": 18.3,
      "width_variance": 5.2,
      "compactness_variance": 3.1
    }
  }
  ```

**Dependencies:**
- Player field coordinates from `player_speeds`

---

#### Task 2.3: Ball Recovery Tracker
**File:** `analytics/recovery_tracker.py`

**Requirements:**
- Detect when team regains possession (possession switches back to them)
- Categorize by field zone:
  - Defensive third (y: 0-22.67m)
  - Midfield (y: 22.67-45.33m)
  - Attacking third (y: 45.33-68m)
- Calculate time since losing possession (recovery time)
- Output:
  ```json
  {
    "team_0": {
      "ball_recoveries": {
        "defensive_third": 15,
        "midfield": 28,
        "attacking_third": 12
      },
      "average_recovery_time_seconds": 3.5,
      "press_recoveries": 8
    }
  }
  ```

**Dependencies:**
- Possession events
- Field coordinates

---

#### Task 2.4: Counter-Attack Detector
**File:** `analytics/counter_attack_detector.py`

**Requirements:**
- Detect rapid possession change (within 2-3 seconds)
- Validate forward movement (team gaining possession moves ball toward opponent goal)
- Track frequency per team
- Output:
  ```json
  {
    "team_0": {
      "counter_attacks": 6,
      "average_counter_attack_distance_m": 35.2,
      "successful_counters": 3
    }
  }
  ```

**Algorithm:**
1. Find possession switches
2. Check time gap between switches (must be < 3 seconds)
3. Calculate ball movement direction (must be forward for gaining team)
4. Validate speed threshold

---

### **Phase 3: Advanced Analytics** (Priority: NICE TO HAVE)

#### Task 3.1: Pressing Intensity Calculator
**File:** `analytics/pressing_calculator.py`

**Requirements:**
- Measure time spent in opponent's half without possession
- Calculate distance covered in defensive third while pressing
- Detect successful presses (ball recovered within X seconds)
- Output pressing intensity index

---

#### Task 3.2: Defensive Line Gap Analyzer
**File:** `analytics/defensive_shape_analyzer.py`

**Requirements:**
- Identify defensive/midfield/forward lines from player positions
- Calculate average gap between lines
- Track gap consistency over time
- Detect defensive shape breaks

---

#### Task 3.3: Movement Synchronization Analyzer
**File:** `analytics/synchronization_analyzer.py`

**Requirements:**
- Analyze correlation of player movements when team doesn't have possession
- Calculate synchronization score (how coordinated team movement is)
- Useful for detecting defensive organization

---

## Completed Features ✅

### **LLM Data Formatter** (Completed)
- ✅ **File:** `analytics/llm_formatter.py`
- ✅ **Integration:** Automatically generates compact JSON after analysis
- ✅ **Output:** `{video}_analysis_data_llm.json`
- ✅ **Size Reduction:** ~95% (from 185K to ~500 lines)
- ✅ **Features:**
  - Match summary with duration, FPS, metadata
  - Match context (teams, field dimensions)
  - Player statistics (performance, possession, passing, shooting)
  - Team statistics (partial - will expand with Phase 1/2 implementation)
  - Match events summary
  - Key insights generation

**Benefits:**
- LLM-friendly: Fits in context windows easily
- Cost-effective: Fewer tokens = lower API costs
- Fast: Smaller payload = faster processing
- Focused: Only relevant data for coaching insights

---

## Implementation Timeline

### **Week 1: Phase 1 - Essential Metrics**
- [ ] Task 1.1: Team Possession Calculator
- [ ] Task 1.2: Pass Completion Rate Calculator
- [ ] Task 1.3: Shots Detection Module
- [ ] Task 1.4: Team Distance Aggregator
- [ ] Task 1.5: Team Analytics Orchestrator
- [ ] Integration with `main.py` pipeline
- [ ] JSON output structure update

### **Week 2: Phase 2 - Tactical Metrics**
- [ ] Task 2.1: Formation Detector
- [ ] Task 2.2: Team Spacing Calculator
- [ ] Task 2.3: Ball Recovery Tracker
- [ ] Task 2.4: Counter-Attack Detector
- [ ] Integration and testing

### **Week 3: Testing & Refinement**
- [ ] Test on sample videos
- [ ] Validate metrics accuracy
- [ ] Refine thresholds and parameters
- [ ] Documentation updates

### **Week 4: LLM Integration Preparation**
- [ ] Finalize JSON schema documentation
- [ ] Create LLM prompt templates
- [ ] Test LLM response quality
- [ ] Iterate based on feedback

---

## Technical Specifications

### **Field Coordinate System**
- Field dimensions: 105m x 68m (standard FIFA)
- Coordinate origin: Bottom-left corner (0, 0)
- X-axis: 0 (left goal) → 105 (right goal)
- Y-axis: 0 (bottom) → 68 (top)
- Goal areas:
  - Left goal: x: 0-5m, y: 25-43m
  - Right goal: x: 100-105m, y: 25-43m

### **Field Zones**
- **Defensive third:** y: 0-22.67m
- **Midfield:** y: 22.67-45.33m
- **Attacking third:** y: 45.33-68m

### **Time Units**
- All durations in seconds
- FPS: 30 (default, configurable)
- Frame-to-second conversion: `seconds = frames / fps`

### **Team Assignment**
- Team 0: Players with IDs 1-11
- Team 1: Players with IDs 12-22
- Referee: ID 0

---

## JSON Output Schema (Target)

```json
{
  "metadata": {
    "processing_time_seconds": 401.27,
    "frames_processed": 750,
    "fps": 30,
    "field_dimensions": {"width": 105, "height": 68}
  },
  "player_analytics": {
    "player_id": {
      "team": 0,
      "average_speed_kmh": 5.2,
      "max_speed_kmh": 18.5,
      "total_distance_m": 2450.3,
      "ball_possession_seconds": 12.5,
      "passes_completed": 8,
      "passes_received": 6,
      "shots_on_goal": 3,
      "shots_total": 5
    }
  },
  "team_analytics": {
    "team_0": {
      "possession": {
        "possession_percentage": 52.3,
        "total_possession_seconds": 450.5
      },
      "passing": {
        "pass_completion_percentage": 78.8,
        "passes_completed": 156,
        "passes_failed": 42,
        "turnovers": 18
      },
      "shooting": {
        "shots_total": 12,
        "shots_on_target": 8
      },
      "distance": {
        "total_distance_km": 45.2,
        "average_per_player_km": 4.1
      },
      "formation": {
        "detected_formation": "4-3-3",
        "formation_confidence": 0.85
      },
      "spacing": {
        "average_width_m": 42.5,
        "average_compactness_m": 18.3
      },
      "recoveries": {
        "defensive_third": 15,
        "midfield": 28,
        "attacking_third": 12,
        "average_recovery_time_seconds": 3.5
      },
      "counter_attacks": 6
    },
    "team_1": {...}
  },
  "player_tracks": {...},
  "ball_tracks": {...},
  "passes": [...],
  "possession_events": [...]
}
```

---

## LLM Coaching Capabilities (After MVP)

Once Phase 1 and Phase 2 are complete, the LLM will be able to:

### **Individual Player Feedback:**
- "How long did Player #5 hold the ball?" → Use `ball_possession_seconds`
- "How many passes did Player #8 complete?" → Use `passes_completed`
- "What was Player #12's average speed?" → Use `average_speed_kmh`
- "How far did Player #3 run?" → Use `total_distance_m`
- "How many shots did Player #9 take?" → Use `shots_total` / `shots_on_goal`

### **Team-Level Tactical Analysis:**
- "Which team had more possession?" → Use `team_analytics[team_X].possession.possession_percentage`
- "What formation is Team 0 playing?" → Use `team_analytics[team_0].formation.detected_formation`
- "How compact was Team 1?" → Use `team_analytics[team_1].spacing.average_compactness_m`
- "How many turnovers did Team 0 have?" → Use `team_analytics[team_0].passing.turnovers`
- "What was Team 1's pass completion rate?" → Use `team_analytics[team_1].passing.pass_completion_percentage`
- "How many counter-attacks did Team 0 execute?" → Use `team_analytics[team_0].counter_attacks`

### **Tactical Suggestions:**
- Formation adjustments based on spacing metrics
- Passing strategy recommendations based on completion rates
- Pressing intensity recommendations
- Defensive shape improvements based on line gaps

---

## Success Criteria for MVP

✅ **Phase 1 Complete When:**
- All individual player metrics working (including shots)
- Team possession % calculated accurately
- Pass completion % calculated accurately
- Turnovers tracked
- Team distance aggregated

✅ **Phase 2 Complete When:**
- Formation detection working with >70% confidence
- Spacing metrics (width/compactness) calculated
- Ball recoveries tracked by zone
- Counter-attacks detected and counted

✅ **MVP Ready When:**
- LLM can answer all individual player questions
- LLM can answer basic team tactical questions
- JSON output is clean and well-structured
- All metrics validated on sample videos

---

## Notes & Considerations

1. **Performance:** Team-level calculations should be efficient. Consider caching intermediate results.

2. **Accuracy:** Some metrics (like formation detection) may need tuning based on real video data.

3. **Edge Cases:** Handle cases where:
   - Players leave field temporarily
   - Ball out of play
   - Camera angles don't capture full field
   - Low-quality keypoint detection

4. **Scalability:** Design metrics to work on videos of any length (30s clips to full matches).

5. **Documentation:** Each new module should include:
   - Docstrings explaining calculation methodology
   - Input/output format specifications
   - Example usage

---

## Next Steps

1. **Review this plan** and prioritize tasks
2. **Start with Phase 1, Task 1.1** (Team Possession Calculator)
3. **Test incrementally** - Validate each metric before moving to next
4. **Iterate based on LLM feedback** - Adjust metrics as you see what works best for coaching insights

---

**Last Updated:** 2024-12-19
**Status:** Ready for Implementation
**Priority:** Phase 1 (Essential Metrics) - CRITICAL for MVP
