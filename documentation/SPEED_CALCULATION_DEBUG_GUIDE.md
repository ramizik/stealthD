# Speed Calculation Debug Guide

## Overview
This document explains the speed calculation implementation and the debug information that has been added to help diagnose speed calculation issues.

## Speed Calculation Flow

### 1. Data Collection
```
Bounding Box (pixels) [x1, y1, x2, y2]
    ↓ [Extract foot position - bottom center: ((x1+x2)/2, y2)]
Pixel Coordinates (x, y)
    ↓
```

### 2. Coordinate Transformation
The system tries three methods in order:

#### Method A: Homography Transformation (Primary)
- Uses detected keypoints to create a homography matrix
- Transforms pixel coordinates to pitch coordinate system (12000 units × 7000 units)
- Scales to real-world meters: **105m × 68m** (standard soccer field)
- Scaling factors:
  - `X_meters = X_pitch_units × (105 / 12000) = X_pitch_units × 0.00875`
  - `Y_meters = Y_pitch_units × (68 / 7000) = Y_pitch_units × 0.00971`

#### Method B: Cached Transformer (Fallback 1)
- If current frame's homography fails, uses last valid transformer
- Assumes camera hasn't moved significantly between frames
- Still applies same scaling factors as Method A

#### Method C: Linear Scaling (Fallback 2)
- **VERY CONSERVATIVE**: Assumes field occupies only central 50% of frame
- Used when no homography is available
- This can UNDERESTIMATE distances, leading to artificially low speeds

### 3. Distance & Speed Calculation
```
Distance (meters) = Euclidean distance between consecutive field coordinates
    ↓ [distance = √((x2-x1)² + (y2-y1)²)]
Time Difference (seconds) = frame_diff / fps
    ↓
Speed (m/s) = distance / time_diff
    ↓ [Convert: × 3.6]
Speed (km/h)
    ↓ [Filter: speeds > 54 km/h are rejected]
Final Speed
```

### 4. Sanity Check Filter
- **Maximum allowed speed**: 54 km/h (15 m/s)
- Rationale: World-class sprinters reach ~43 km/h, with margin for measurement
- Speeds exceeding this are filtered out as tracking/transformation errors

---

## Debug Output Interpretation

When you run the pipeline, you'll see detailed debug output like this:

### Section 1: Initial Configuration
```
🔍 [SPEED DEBUG] Starting speed calculation:
   Video dimensions: (1920, 1080)
   Field dimensions: 105.0m x 68.0m
   FPS: 30.0
   Total frames with transformers: 1500
```

**What to check:**
- ✅ Field dimensions should be 105m × 68m (standard soccer field)
- ✅ FPS should match your video (typically 24, 25, or 30)
- ✅ Video dimensions should match your input video

### Section 2: Transformation Method Statistics
```
🔍 [SPEED DEBUG] Transformation Method Statistics:
   ✓ Homography successful: 8542
   ✗ Homography failed: 245
   ⚠ Catastrophic failures: 12
   📦 Cached transformer used: 1523
   📏 Linear fallback used: 78
```

**What to check:**
- ✅ **Homography successful**: High percentage is good (>70%)
- ⚠️ **Catastrophic failures**: Should be minimal (<5%)
  - If high: Keypoint detection is failing, check your keypoint model
- ⚠️ **Linear fallback used**: Should be rare (<10%)
  - If high: Many frames have no valid homography, speeds will be underestimated

### Section 3: Top 5 Fastest Players
```
🔍 [SPEED DEBUG] Top 5 Fastest Players:
   1. Player 7: 32.45 km/h (avg: 8.32 km/h)
      └─ Max speed: 32.45 km/h (✓ accepted)
      └─ Distance: 8.95m over 1 frames (0.033s)
      └─ From frame 1245 to 1246
      └─ Position: (52.34, 28.67) → (61.29, 28.89) meters
      └─ Transform method: homography
```

**What to check for each player:**

1. **Max Speed Value**
   - ✅ 25-35 km/h: Realistic sprint speed for soccer
   - ⚠️ 40-50 km/h: Possible but rare (world-class sprint)
   - 🚨 >50 km/h: Error (should be filtered, but check if getting through)

2. **Distance and Time**
   - Calculate: `speed = (distance / time) × 3.6`
   - Verify the math matches the reported speed
   - Example: `(8.95m / 0.033s) × 3.6 = 32.45 km/h` ✓

3. **Frame Difference**
   - ⚠️ If `frame_diff > 1`: Tracking lost player, speed might be unreliable
   - ✅ `frame_diff = 1`: Continuous tracking, most reliable

4. **Position Change**
   - Check if the position change makes sense
   - Example: (52.34, 28.67) → (61.29, 28.89) = 8.95m movement
   - Calculate: `√((61.29-52.34)² + (28.89-28.67)²) = 8.95m` ✓

5. **Transform Method**
   - ✅ **homography**: Most accurate, uses keypoint detection
   - ⚠️ **cached**: Acceptable, using recent homography
   - 🚨 **linear_fallback**: Least accurate, may underestimate

6. **Filtered Status**
   - ✅ `✓ accepted`: Speed passed sanity check
   - 🚨 `⚠️ FILTERED`: Speed exceeded 54 km/h limit
     - If you see filtered speeds, investigate why they're so high

---

## Common Issues and Diagnosis

### Issue 1: Speeds are consistently ~50 km/h (TOO HIGH)

**Possible Causes:**
1. **Incorrect pitch coordinate scaling**
   - Check if ViewTransformer returns units other than 12000×7000
   - Verify scaling factors in `calculate_field_coordinates()`

2. **Frame skipping**
   - If `frame_diff` is often >1, time calculation is wrong
   - Check tracking continuity

3. **Homography errors**
   - High "catastrophic failures" count
   - Bad keypoint detection producing wrong transformations
   - Solution: Improve keypoint model or detection confidence threshold

4. **Wrong FPS**
   - If FPS is wrong, time calculations are wrong
   - Verify FPS matches video

### Issue 2: Speeds are too LOW (<20 km/h for sprints)

**Possible Causes:**
1. **Linear fallback overuse**
   - Conservative 50% field assumption underestimates
   - Check "Linear fallback used" count
   - Solution: Improve homography success rate

2. **Field dimension mismatch**
   - If field dimensions are wrong, scaling is wrong
   - Verify 105m × 68m

### Issue 3: Speeds vary wildly frame-to-frame

**Possible Causes:**
1. **Tracking instability**
   - Player IDs switching between detections
   - Check tracking pipeline

2. **Homography instability**
   - Transformations varying between frames
   - Check keypoint detection consistency

---

## Debugging Steps

### Step 1: Run the pipeline
```bash
python main.py
```

### Step 2: Review CLI output
Look for the debug sections:
- Initial configuration
- Transformation statistics
- Top 5 fastest players

### Step 3: Check the JSON output
Open `input_videos/sample_1_analysis_data.json` and examine:
- `player_speeds[player_id].max_speed_kmh`
- `player_speeds[player_id].field_coordinates`
- Verify coordinates are within field bounds (0-105m, 0-68m)

### Step 4: Analyze specific samples
For the top 5 players, manually verify:
1. Calculate speed from distance and time
2. Check if position change makes sense
3. Verify transformation method used

### Step 5: Check for patterns
- Are all high speeds using the same transformation method?
- Are high speeds occurring at specific frame ranges?
- Is there correlation with camera movement?

---

## Expected Values

### Soccer Player Speeds (Reference)
- **Walking**: 3-7 km/h
- **Jogging**: 8-15 km/h
- **Running**: 15-25 km/h
- **Sprinting**: 25-35 km/h
- **Elite sprint**: 35-38 km/h (world-class)
- **Maximum human sprint**: ~43-44 km/h (Usain Bolt)

### System Limits
- **Filter threshold**: 54 km/h (15 m/s)
- **Field dimensions**: 105m × 68m
- **Pitch coordinate system**: 12000 × 7000 units

---

## Next Steps

After reviewing the debug output:

1. **If speeds are accurate**: Great! The system is working correctly.

2. **If speeds are too high**:
   - Check pitch coordinate scaling factors
   - Verify homography transformation output units
   - Check FPS accuracy
   - Review keypoint detection quality

3. **If speeds are too low**:
   - Increase homography success rate
   - Check field dimension configuration
   - Verify tracking continuity

4. **If you need more detail**:
   - Add logging to `calculate_field_coordinates()` to see raw transformation outputs
   - Log pixel coordinates vs field coordinates for sample frames
   - Track homography matrix values

---

## Code Locations

- **Speed Calculator**: `analytics/speed_calculator.py`
- **Metrics Calculator**: `analytics/metrics_calculator.py`
- **Homography Transformer**: `tactical_analysis/homography.py`
- **Main Pipeline**: `main.py`
- **Constants**: `constants.py`
