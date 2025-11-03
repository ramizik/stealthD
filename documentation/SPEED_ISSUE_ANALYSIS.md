# Speed Calculation Issue Analysis

## Executive Summary

**PROBLEM**: Player speeds are consistently maxing out at ~54 km/h, which is unrealistic for soccer players.

**ROOT CAUSE**: Homography transformation is producing catastrophically incorrect field coordinates, leading to:
- Extreme out-of-bounds positions (10.9% of all coordinates are >10m outside the field)
- Impossibly large distances between frames (mean: 146.56m per frame)
- Speeds that are filtered down to the 54 km/h sanity check limit

**STATUS**: The sanity check IS working (filtering speeds > 54 km/h), but the underlying coordinate transformation is fundamentally broken.

---

## Diagnostic Results (from existing data)

### Critical Issues Found

#### 1. Field Coordinate Errors (CRITICAL)
```
Total coordinates: 14,317
Out of bounds: 2,461 (17.2%)
Extreme out of bounds (>10m): 1,560 (10.9%)
```

**Examples of Bad Coordinates:**
- Player 4, Frame 101: Position at `(6.15, -104.66)` meters
  - Y-coordinate is **104 meters BELOW** the field bottom!
  - Field should be 0-68m in Y-axis

- Player 4, Frames 66-70: All positions with Y > 80m
  - Field maximum Y should be 68m
  - These are 12-24m above the field

#### 2. Distance Anomalies (CRITICAL)
```
Mean distance per frame: 146.56 m
Median distance per frame: 0.16 m
Max distance: 336,882.07 m (!)
95th percentile: 7.29 m
```

**Analysis:**
- **Median (0.16m)** is correct for typical player movement
- **Mean (146.56m)** is skewed by extreme outliers
- At 25 fps, elite sprint speed (~10 m/s) should produce **0.4m per frame**
- Seeing distances of **30+ meters per frame** means transformation is wrong

**Examples:**
- Player 2: 32.35m in 1 frame
  - This equals 809 m/s = 2,912 km/h (before filtering!)
- Player 4: 28.06m in 1 frame
  - Position jump: (6.15, -104.66) → (16.62, -78.62)

#### 3. Speed Distribution (CONSEQUENCE)
```
Suspicious speeds (>44 km/h): 476 measurements (4.6%)
Mean max speed: 53.13 km/h
Median max speed: 53.62 km/h
Highest speed: 53.97 km/h
```

**Top 10 Players - ALL have speeds ~54 km/h:**
1. Player 5: 53.97 km/h
2. Player 6: 53.96 km/h
3. Player 13: 53.96 km/h
4. Player 7: 53.95 km/h
5. Player 21: 53.95 km/h
...

**Why this pattern?**
- The sanity check filter caps speeds at 54 km/h (15 m/s)
- Many calculated speeds exceed this limit and get filtered
- The remaining speeds cluster RIGHT at the limit
- This proves the underlying calculations are producing even higher speeds

---

## Root Cause Analysis

### The Calculation Chain

```
Frame pixels → Homography Transform → Field coordinates → Distance → Speed
```

**Where it breaks: Homography Transform**

### Homography Transformation System

The system uses keypoint detection to establish field geometry:
1. Detects 29 keypoints on the field (lines, circles, corners)
2. Maps detected points to known pitch coordinates (12000×7000 unit system)
3. Creates transformation matrix to convert pixel coordinates to field meters
4. Scales: X (12000 units → 105m), Y (7000 units → 68m)

**Current Status:**
- 10.9% of transformations produce extreme failures
- Coordinates ending up 100+ meters off the field
- This indicates keypoint detection quality issues

### Why Coordinates Go Wrong

**Scenario A: Keypoint Detection Fails**
- Not enough keypoints detected (need minimum 4)
- Low confidence keypoints included
- Wrong keypoints matched to field positions

**Scenario B: Homography Matrix is Degenerate**
- Keypoints don't provide enough geometric constraints
- Results in unstable transformation
- Small pixel errors amplified to huge field coordinate errors

**Scenario C: Linear Fallback Overused**
- When homography fails, system uses "conservative linear scaling"
- Assumes field occupies only central 50% of frame
- This can underestimate OR overestimate depending on actual field coverage

### Example Failure Case

**Player 4, Frame 101-102:**
```
Frame 101 pixel position: (x₁, y₁)
  ↓ [Bad homography transformation]
Frame 101 field position: (6.15, -104.66) meters  ← Y is 104m BELOW field!

Frame 102 pixel position: (x₂, y₂)
  ↓ [Another bad transformation]
Frame 102 field position: (16.62, -78.62) meters  ← Y is 78m BELOW field!

Distance = √[(16.62-6.15)² + (-78.62-(-104.66))²] = 28.06 meters in 1 frame

Speed = 28.06m / (1/25)s × 3.6 = 2,525 km/h
  ↓ [Filtered by sanity check]
Speed recorded = REJECTED (> 54 km/h limit)
```

---

## Why Speeds Cluster at ~54 km/h

The sanity check in the code:
```python
if instant_speed > 15:  # 15 m/s = 54 km/h
    continue  # Skip this measurement
```

This means:
1. Bad transformations produce speeds of 100-2000+ km/h
2. These get filtered out completely
3. Only speeds below 54 km/h are kept
4. But many "real" speeds are inflated to 40-53 km/h by smaller transformation errors
5. Result: Top speeds cluster just below the filter threshold

---

## Solutions (In Order of Priority)

### Solution 1: Improve Keypoint Detection (PRIMARY FIX)

**Increase confidence threshold:**
```python
# In tactical_analysis/homography.py
def __init__(self, confidence_threshold=0.5):  # Currently 0.5
```

Try increasing to 0.6 or 0.7 to only use high-confidence keypoints.

**Verify keypoint model quality:**
- Check keypoint detection model accuracy
- Ensure sufficient training data quality
- Consider retraining if detection rate is low

**Add keypoint validation:**
- Reject transformations if keypoints are geometrically inconsistent
- Check if detected field geometry matches expected ratios

### Solution 2: Strengthen Catastrophic Failure Detection (IMMEDIATE FIX)

**Current threshold is too lenient:**
```python
# In analytics/speed_calculator.py, line 56
x_bad = np.any((field_coords[:, 0] < -10) | (field_coords[:, 0] > self.field_width + 10))
y_bad = np.any((field_coords[:, 1] < -10) | (field_coords[:, 1] > self.field_height + 10))
```

**Recommended: Tighten to ±5m:**
```python
x_bad = np.any((field_coords[:, 0] < -5) | (field_coords[:, 0] > self.field_width + 5))
y_bad = np.any((field_coords[:, 1] < -5) | (field_coords[:, 1] > self.field_height + 5))
```

This will force more use of cached transformers and reject more bad homographies.

### Solution 3: Improve Transformer Caching Strategy

**Current approach:**
- Cache last valid transformer
- Use it for frames with failed homography

**Enhancement:**
- Keep a sliding window of N recent valid transformers
- Use temporal interpolation between nearby valid transformers
- Track camera movement confidence

### Solution 4: Add Multi-Frame Speed Smoothing

**Current: Instant speed frame-to-frame**
- Very sensitive to single-frame errors

**Better: Moving average or Kalman filter**
```python
# Calculate speed over 3-5 frame windows
# Smooth outliers
# Use median instead of mean for robustness
```

### Solution 5: Alternative Coordinate Validation

**Add physics-based validation:**
- Maximum acceleration check (not just speed)
- Human acceleration limit: ~4 m/s²
- Flag sudden direction changes >90°

**Distance-based sanity check:**
- If distance > 2 meters in one frame (at 25 fps)
  - That's 50 m/s = 180 km/h
  - Reject the transformation for that frame

---

## Recommended Action Plan

### Phase 1: Immediate Fixes (Can implement now)

1. **Tighten catastrophic failure threshold** to ±5m
   - File: `analytics/speed_calculator.py`, line 56

2. **Add distance-based validation**
   - Before calculating speed, check if distance > 2m in one frame
   - If yes, skip this speed measurement

3. **Lower speed filter threshold** to 45 km/h (12.5 m/s)
   - More realistic for soccer players
   - Reduces false positives

### Phase 2: Diagnostic Analysis (Next run)

Run the pipeline with new debug output to observe:
1. How often homography fails vs succeeds
2. How often cached transformer is used
3. How often linear fallback is used
4. Correlation between transformation method and high speeds

### Phase 3: Long-term Improvements

1. **Improve keypoint detection**
   - Increase confidence threshold
   - Validate keypoint model quality
   - Consider ensemble or multi-frame keypoint detection

2. **Implement multi-frame smoothing**
   - Moving average filter for speeds
   - Kalman filter for position tracking
   - Outlier rejection using statistical methods

3. **Add homography validation**
   - Check field geometry ratios
   - Validate transformation matrix properties
   - Reject geometrically impossible transformations

---

## Testing & Validation

### After implementing fixes, verify:

1. **Field coordinate bounds:**
   - <5% should be out of bounds
   - <1% should be extreme out of bounds

2. **Distance distribution:**
   - Mean should be <1m per frame
   - 99th percentile should be <0.6m per frame (at 25 fps)

3. **Speed distribution:**
   - Max speeds should be 25-35 km/h for most players
   - Only 1-2 elite players might reach 35-40 km/h
   - No speeds should exceed 45 km/h regularly

4. **Transformation success rate:**
   - >80% homography success
   - <10% cached transformer usage
   - <5% linear fallback usage
   - <1% catastrophic failures

---

## Debug Commands

### Analyze existing JSON output:
```bash
python analytics/speed_diagnostic.py input_videos/sample_1_analysis_data.json
```

### Run pipeline with new debug output:
```bash
python main.py
```

Look for:
```
🔍 [SPEED DEBUG] Starting speed calculation:
   Video dimensions: (width, height)
   Field dimensions: 105.0m x 68.0m
   FPS: 25.0
   Total frames with transformers: N

🔍 [SPEED DEBUG] Transformation Method Statistics:
   ✓ Homography successful: X
   ✗ Homography failed: Y
   ⚠ Catastrophic failures: Z
   📦 Cached transformer used: A
   📏 Linear fallback used: B

🔍 [SPEED DEBUG] Top 5 Fastest Players:
   [Detailed breakdown of max speed calculations]
```

---

## References

- **Speed Calculator**: `analytics/speed_calculator.py`
- **Homography Transformer**: `tactical_analysis/homography.py`
- **Keypoint Detection**: `keypoint_detection/detect_keypoints.py`
- **Debug Guide**: `documentation/SPEED_CALCULATION_DEBUG_GUIDE.md`
- **Diagnostic Tool**: `analytics/speed_diagnostic.py`
