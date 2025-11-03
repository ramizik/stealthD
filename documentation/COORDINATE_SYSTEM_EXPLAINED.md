# Soccer Field Coordinate System - Complete Explanation

## Overview

Your system uses a **two-stage coordinate system** for tracking players on the field.

---

## Stage 1: Pitch Coordinate System (Internal Units)

### Dimensions
```
Length (X-axis): 12,000 units = 105 meters
Width (Y-axis):  7,000 units = 68 meters
```

### Origin and Orientation
```
(0, 0) ──────────────────────── (12000, 0)
  │    TOP OF FIELD              │
  │                               │
  │         X-AXIS                │
  │      (Horizontal)             │
  │   Left Goal → Right Goal      │
  │                               │
  │                               │
  │       Y-AXIS                  │
  │      (Vertical)               │
  │    Top → Bottom               │
  │                               │
(0, 7000) ──────────────────── (12000, 7000)
         BOTTOM OF FIELD
```

**Key Points:**
- **Origin (0, 0)**: Top-left corner of the field
- **X-axis**: Horizontal, 0 (left goal) → 12,000 (right goal)
- **Y-axis**: Vertical, 0 (top sideline) → 7,000 (bottom sideline)
- **Center of field**: (6000, 3500) units

### Field Landmarks in Pitch Coordinates

```python
# Corner points
Top-left corner:     [0, 0]
Top-right corner:    [12000, 0]
Bottom-left corner:  [0, 7000]
Bottom-right corner: [12000, 7000]

# Center
Center point:        [6000, 3500]
Center line top:     [6000, 0]
Center line bottom:  [6000, 7000]

# Left penalty area (near [0, y])
Left penalty top:    [1980, 1570]  # 1570 = (7000 - 4860) / 2
Left penalty bottom: [1980, 5430]  # 5430 = (7000 + 4860) / 2

# Right penalty area (near [12000, y])
Right penalty top:    [10020, 1570]
Right penalty bottom: [10020, 5430]
```

---

## Stage 2: Real-World Coordinates (Meters)

### Conversion Formula

The pitch units are converted to real-world meters using:

```python
X_meters = X_pitch_units × (105 / 12000) = X_pitch_units × 0.00875
Y_meters = Y_pitch_units × (68 / 7000)   = Y_pitch_units × 0.00971428
```

### Valid Ranges

**After conversion to meters:**
```
X-axis: 0.0 to 105.0 meters  (left to right)
Y-axis: 0.0 to 68.0 meters   (top to bottom)
```

**Example conversions:**
```
Pitch (0, 0)         → (0.00, 0.00) meters    [Top-left corner]
Pitch (12000, 7000)  → (105.0, 68.0) meters   [Bottom-right corner]
Pitch (6000, 3500)   → (52.5, 34.0) meters    [Center]
Pitch (1980, 3500)   → (17.33, 34.0) meters   [Left penalty spot area]
```

---

## Coordinate System Validation Rules

### ✅ Valid Coordinates

A coordinate is **valid** if:
```
0 ≤ X ≤ 105 meters
0 ≤ Y ≤ 68 meters
```

### ⚠️ Acceptable Out-of-Bounds (Edge Distortion)

Minor out-of-bounds is acceptable due to:
- Camera perspective distortion
- Keypoint detection uncertainty
- Players near field edges

**Tolerance: ±5 meters**
```
-5 ≤ X ≤ 110 meters  (acceptable)
-5 ≤ Y ≤ 73 meters   (acceptable)
```

### 🚨 Catastrophically Bad Coordinates

Coordinates are **catastrophically wrong** if:
```
X < -10 meters  OR  X > 115 meters
Y < -10 meters  OR  Y > 78 meters
```

These indicate homography transformation failure.

---

## Your Data Analysis Results

From the diagnostic tool on `sample_1_analysis_data.json`:

### Examples of Bad Coordinates

```
Player 4, Frame 66:  (83.45, 80.87) meters
   ❌ Y = 80.87m > 68m (12.87m above field)

Player 4, Frame 67:  (86.76, 87.42) meters
   ❌ Y = 87.42m > 68m (19.42m above field)

Player 4, Frame 101: (6.15, -104.66) meters
   ❌ Y = -104.66m < 0m (104.66m BELOW field!)
```

### Why Negative Y Coordinates Are Impossible

The Y-axis origin is at the **TOP** of the field:
```
Y = 0     ← TOP sideline (valid)
Y = 34    ← Center of field (valid)
Y = 68    ← BOTTOM sideline (valid)
Y = -10   ← 10 meters ABOVE the top sideline (INVALID)
Y = -104  ← 104 meters ABOVE the top sideline (CATASTROPHIC ERROR)
```

**A negative Y coordinate means the transformation thinks the player is ABOVE the top sideline**, which is geometrically impossible for a player on the field.

---

## Transformation Pipeline

### Step 1: Pixel Coordinates (Frame)
```
Camera view: (pixel_x, pixel_y)
Example: (854, 623) in a 1920x1080 video
```

### Step 2: Homography Transformation
```python
# Detect 29 keypoints on the field
keypoints_frame = detect_keypoints(frame)
# Examples: corner flags, penalty box corners, center circle

# Map detected keypoints to known pitch coordinates
# Example: Top-left corner detected at pixel (120, 85)
#          Maps to pitch coordinate (0, 0)

# Create transformation matrix using OpenCV
matrix, mask = cv2.findHomography(frame_points, pitch_points, cv2.RANSAC, 5.0)

# Transform player position
pitch_coords = cv2.perspectiveTransform(pixel_coords, matrix)
# Result: (x_pitch, y_pitch) in 0-12000, 0-7000 units
```

### Step 3: Scale to Meters
```python
# Convert pitch units to real-world meters
x_meters = x_pitch × (105.0 / 12000.0)
y_meters = y_pitch × (68.0 / 7000.0)

# Result: (x_meters, y_meters) in 0-105m, 0-68m range
```

### Step 4: Validation
```python
# Check if catastrophically bad
if x_meters < -10 or x_meters > 115:
    # BAD: Use cached transformer or linear fallback
if y_meters < -10 or y_meters > 78:
    # BAD: Use cached transformer or linear fallback
```

---

## Where Things Go Wrong

### Problem 1: Insufficient Keypoints
```
Required: Minimum 4 keypoints with confidence > 0.5
Problem: Only 2-3 detected, or low confidence
Result: Cannot create valid homography matrix
```

### Problem 2: Incorrect Keypoint Matching
```
Example Error:
  Detected keypoint at pixel (856, 234)
  Incorrectly identified as "top-left corner"
  Actually should be "center circle top"

Result: Wrong mapping → Wrong matrix → Wrong coordinates
```

### Problem 3: Degenerate Homography Matrix
```
Problem: Keypoints are nearly collinear (on a line)
Result: Matrix becomes unstable
Effect: Small pixel errors → Huge coordinate errors
```

### Problem 4: Camera Angle Extremes
```
Problem: Very angled camera view (not top-down)
Result: Perspective distortion too severe
Effect: Edge coordinates become unreliable
```

---

## Debug Example: Player 4, Frame 101

### What Happened
```
Input: Player 4 foot position in video frame
       (Unknown exact pixel, but somewhere in 1920x1080 frame)

Homography Transform:
       Pixel → Pitch units
       Result: Unknown exact pitch units, but maps to very negative Y

Scale to Meters:
       Pitch units × (68/7000) = -104.66 meters

ERROR: Y = -104.66m means:
       - 104.66 meters ABOVE the top sideline
       - Player would be floating in the air outside stadium!
       - Homography matrix is completely wrong for this frame
```

### Why This Happened
```
Most Likely Causes:
1. Keypoints not detected in frame 101
2. Only 2-3 keypoints detected (insufficient)
3. Wrong keypoints matched to field positions
4. Camera angle made transformation unstable

System Response:
- Speed calculation uses this bad coordinate
- Distance from frame 100→101 becomes huge
- Speed calculation: huge distance / time = 1000+ km/h
- Sanity check filters it out (> 54 km/h)
```

---

## Validation in Your Code

### Current Implementation (speed_calculator.py)

```python
def _is_catastrophically_bad(self, field_coords: np.ndarray) -> bool:
    """Check if coordinates are catastrophically bad."""
    if len(field_coords) == 0:
        return False

    # Reject if any coordinate is MORE than 10m outside field bounds
    x_bad = np.any((field_coords[:, 0] < -10) |
                   (field_coords[:, 0] > self.field_width + 10))
    y_bad = np.any((field_coords[:, 1] < -10) |
                   (field_coords[:, 1] > self.field_height + 10))

    return x_bad or y_bad
```

**Current threshold: ±10 meters**
- Allows Y from -10 to 78 meters
- Player at Y=-104.66m is caught as catastrophic ✓
- But Y=80.87m (12.87m over) is NOT caught ✗

### Recommended Fix

```python
def _is_catastrophically_bad(self, field_coords: np.ndarray) -> bool:
    """Check if coordinates are catastrophically bad."""
    if len(field_coords) == 0:
        return False

    # Tighter threshold: ±5 meters
    x_bad = np.any((field_coords[:, 0] < -5) |
                   (field_coords[:, 0] > self.field_width + 5))
    y_bad = np.any((field_coords[:, 1] < -5) |
                   (field_coords[:, 1] > self.field_height + 5))

    return x_bad or y_bad
```

**Improved threshold: ±5 meters**
- Allows Y from -5 to 73 meters
- Player at Y=80.87m (12.87m over) IS caught ✓
- Player at Y=-104.66m is caught ✓

---

## Coordinate System Summary

### Quick Reference
```
Pitch Units → Meters Conversion:
  X: 12,000 units = 105 meters  (0.00875 m/unit)
  Y: 7,000 units = 68 meters    (0.00971 m/unit)

Origin: (0, 0) = Top-left corner
X-axis: Left → Right (0 to 105m)
Y-axis: Top → Bottom (0 to 68m)

Valid ranges:
  X: 0 to 105 meters
  Y: 0 to 68 meters

Acceptable tolerance: ±5 meters
Catastrophic threshold: ±10 meters (current), should be ±5 meters
```

### Key Field Positions
```
Top-left corner:     (0, 0)
Top-right corner:    (105, 0)
Bottom-left corner:  (0, 68)
Bottom-right corner: (105, 68)
Center:              (52.5, 34)

Left penalty spot:   (~11, 34)
Right penalty spot:  (~94, 34)
```

### Coordinate Interpretation
```
Player at (30, 15) meters:
  - 30m from left goal (28% across field)
  - 15m from top sideline (22% down field)
  - In left third of field, closer to top

Player at (80, 50) meters:
  - 80m from left goal (76% across field)
  - 50m from top sideline (74% down field)
  - In right third of field, closer to bottom

Player at (52.5, 34) meters:
  - Exactly at center of field
```

---

## Testing Your Understanding

### Question: What do these coordinates mean?

```
1. Player at (0, 34) meters
   Answer: Left goal line, center height
   (On the goal line on the left side)

2. Player at (105, 0) meters
   Answer: Top-right corner flag
   (Right corner, top sideline)

3. Player at (52.5, 0) meters
   Answer: Center of top sideline
   (Halfway line at top edge)

4. Player at (-10, 34) meters
   Answer: 10 meters behind left goal
   (Outside field, behind left goal - should be caught as bad)

5. Player at (52.5, -20) meters
   Answer: 20 meters above top sideline
   (Impossible - should be caught as catastrophic)

6. Player at (52.5, 100) meters
   Answer: 32 meters below bottom sideline
   (Impossible - should be caught as catastrophic)
```

---

## Files to Check

### Coordinate System Definition
- **`tactical_analysis/sports_compat.py`** (lines 78-170)
  - `SoccerPitchConfiguration.__init__()`: Defines 12000×7000 units
  - `_create_pitch_vertices()`: All 35 reference points

### Keypoint Mapping
- **`tactical_analysis/homography.py`** (lines 30-72)
  - `_get_all_pitch_points()`: Gets pitch coordinate reference
  - `_get_keypoint_mapping()`: Maps 29 detected keypoints to pitch points

### Transformation
- **`tactical_analysis/sports_compat.py`** (lines 41-63)
  - `ViewTransformer.transform_points()`: Does the homography transformation

### Scaling to Meters
- **`analytics/speed_calculator.py`** (lines 124-130)
  - Applies (105/12000) and (68/7000) scaling factors

### Validation
- **`analytics/speed_calculator.py`** (lines 39-59)
  - `_is_catastrophically_bad()`: Checks if coordinates are valid

---

## Summary

Your coordinate system is **correctly designed** but the **homography transformation is failing** for some frames, producing:

1. ❌ Negative Y coordinates (impossible - above top sideline)
2. ❌ Y > 100 meters (impossible - far below bottom sideline)
3. ❌ Extreme X coordinates outside field

**The fix is NOT to change the coordinate system** - it's working as designed.

**The fix IS to:**
1. Tighten validation thresholds (±5m instead of ±10m)
2. Improve keypoint detection quality
3. Add distance-based validation
4. Better handle frames with failed homography

The coordinate system itself (origin at top-left, Y increasing downward) is standard and correct. The problem is the transformation FROM pixels TO this coordinate system is breaking down.
