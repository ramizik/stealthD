#!/usr/bin/env python3
"""
Speed Calculation Diagnostic Tool

Analyzes speed data from analysis JSON to help diagnose speed calculation issues.
Can be run on existing JSON output without reprocessing video.

Usage:
    python analytics/speed_diagnostic.py input_videos/sample_1_analysis_data.json
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

import json
import numpy as np
from typing import Dict, List, Tuple
import argparse


def analyze_field_coordinates(player_speeds: Dict, field_width: float = 105, field_height: float = 68):
    """Analyze field coordinates for out-of-bounds or unusual values."""

    print("\n" + "="*70)
    print("FIELD COORDINATE ANALYSIS")
    print("="*70)

    out_of_bounds_count = 0
    extreme_out_of_bounds_count = 0
    total_coords = 0

    for player_id, data in player_speeds.items():
        field_coords = data.get('field_coordinates', {})

        for frame_idx, coord in field_coords.items():
            x, y = coord
            total_coords += 1

            # Check if out of bounds
            if x < 0 or x > field_width or y < 0 or y > field_height:
                out_of_bounds_count += 1

                # Check if extremely out of bounds (>10m)
                if x < -10 or x > field_width + 10 or y < -10 or y > field_height + 10:
                    extreme_out_of_bounds_count += 1
                    if extreme_out_of_bounds_count <= 5:  # Show first 5 examples
                        print(f"[!] Player {player_id}, Frame {frame_idx}: ({x:.2f}, {y:.2f}) - "
                              f"EXTREME out of bounds")

    print(f"\nTotal coordinates: {total_coords}")
    print(f"Out of bounds: {out_of_bounds_count} ({100*out_of_bounds_count/total_coords:.1f}%)")
    print(f"Extreme out of bounds (>10m): {extreme_out_of_bounds_count} ({100*extreme_out_of_bounds_count/total_coords:.1f}%)")

    if extreme_out_of_bounds_count > total_coords * 0.05:
        print("\n[!!] WARNING: >5% of coordinates are extremely out of bounds!")
        print("   This suggests homography transformation errors.")


def analyze_speed_distribution(player_speeds: Dict):
    """Analyze the distribution of speeds."""

    print("\n" + "="*70)
    print("SPEED DISTRIBUTION ANALYSIS")
    print("="*70)

    all_speeds = []
    max_speeds = []
    avg_speeds = []

    for player_id, data in player_speeds.items():
        max_speeds.append(data.get('max_speed_kmh', 0))
        avg_speeds.append(data.get('average_speed_kmh', 0))

        frame_speeds = data.get('frames', {})
        all_speeds.extend(frame_speeds.values())

    if not all_speeds:
        print("No speed data found!")
        return

    all_speeds = np.array(all_speeds)
    max_speeds = np.array(max_speeds)
    avg_speeds = np.array(avg_speeds)

    print("\n[STATS] All Instantaneous Speeds:")
    print(f"   Count: {len(all_speeds)}")
    print(f"   Mean: {np.mean(all_speeds):.2f} km/h")
    print(f"   Median: {np.median(all_speeds):.2f} km/h")
    print(f"   Std Dev: {np.std(all_speeds):.2f} km/h")
    print(f"   Min: {np.min(all_speeds):.2f} km/h")
    print(f"   Max: {np.max(all_speeds):.2f} km/h")
    print(f"   95th percentile: {np.percentile(all_speeds, 95):.2f} km/h")
    print(f"   99th percentile: {np.percentile(all_speeds, 99):.2f} km/h")

    # Speed ranges
    walking = np.sum((all_speeds >= 0) & (all_speeds < 7))
    jogging = np.sum((all_speeds >= 7) & (all_speeds < 15))
    running = np.sum((all_speeds >= 15) & (all_speeds < 25))
    sprinting = np.sum((all_speeds >= 25) & (all_speeds < 35))
    elite = np.sum((all_speeds >= 35) & (all_speeds < 44))
    suspicious = np.sum(all_speeds >= 44)

    print("\n[DISTRIBUTION] Speed Categories:")
    print(f"   Walking (0-7 km/h): {walking} ({100*walking/len(all_speeds):.1f}%)")
    print(f"   Jogging (7-15 km/h): {jogging} ({100*jogging/len(all_speeds):.1f}%)")
    print(f"   Running (15-25 km/h): {running} ({100*running/len(all_speeds):.1f}%)")
    print(f"   Sprinting (25-35 km/h): {sprinting} ({100*sprinting/len(all_speeds):.1f}%)")
    print(f"   Elite Sprint (35-44 km/h): {elite} ({100*elite/len(all_speeds):.1f}%)")
    print(f"   [!!] Suspicious (>44 km/h): {suspicious} ({100*suspicious/len(all_speeds):.1f}%)")

    if suspicious > 0:
        print(f"\n[!] WARNING: {suspicious} speed measurements exceed realistic human sprint speed!")
        print("   These should have been filtered by the 54 km/h sanity check.")

    print("\n[STATS] Player Max Speeds:")
    print(f"   Mean: {np.mean(max_speeds):.2f} km/h")
    print(f"   Median: {np.median(max_speeds):.2f} km/h")
    print(f"   Highest: {np.max(max_speeds):.2f} km/h")


def analyze_distances(player_speeds: Dict):
    """Analyze movement distances frame-to-frame."""

    print("\n" + "="*70)
    print("DISTANCE ANALYSIS")
    print("="*70)

    all_distances = []
    suspicious_distances = []

    for player_id, data in player_speeds.items():
        field_coords = data.get('field_coordinates', {})

        if len(field_coords) < 2:
            continue

        # Sort by frame index
        sorted_frames = sorted(field_coords.items())

        for i in range(1, len(sorted_frames)):
            prev_frame, prev_coord = sorted_frames[i-1]
            curr_frame, curr_coord = sorted_frames[i]

            # Calculate distance
            distance = np.sqrt((curr_coord[0] - prev_coord[0])**2 +
                             (curr_coord[1] - prev_coord[1])**2)

            frame_diff = int(curr_frame) - int(prev_frame)

            # Only consider consecutive frames for this analysis
            if frame_diff == 1:
                all_distances.append(distance)

                # Suspicious: >10m in one frame at 30fps = >1080 km/h
                if distance > 10:
                    suspicious_distances.append({
                        'player_id': player_id,
                        'from_frame': prev_frame,
                        'to_frame': curr_frame,
                        'distance': distance,
                        'from_pos': prev_coord,
                        'to_pos': curr_coord
                    })

    if not all_distances:
        print("No distance data found!")
        return

    all_distances = np.array(all_distances)

    print("\n[DISTANCE] Frame-to-Frame Distances (consecutive frames only):")
    print(f"   Count: {len(all_distances)}")
    print(f"   Mean: {np.mean(all_distances):.2f} m")
    print(f"   Median: {np.median(all_distances):.2f} m")
    print(f"   Max: {np.max(all_distances):.2f} m")
    print(f"   95th percentile: {np.percentile(all_distances, 95):.2f} m")
    print(f"   99th percentile: {np.percentile(all_distances, 99):.2f} m")

    # At 30 fps, 1 frame = 0.033s
    # Elite sprint ~10 m/s = 0.33m per frame
    # Max human sprint ~12 m/s = 0.4m per frame
    suspicious_count = np.sum(all_distances > 0.5)
    extreme_count = np.sum(all_distances > 1.0)

    print(f"\n[!] Suspicious distances (>0.5m in 1 frame): {suspicious_count} ({100*suspicious_count/len(all_distances):.1f}%)")
    print(f"[!!] Extreme distances (>1.0m in 1 frame): {extreme_count} ({100*extreme_count/len(all_distances):.1f}%)")

    if extreme_count > 0:
        print("\n[!!] EXTREME DISTANCE EXAMPLES (first 5):")
        for i, example in enumerate(suspicious_distances[:5]):
            print(f"   Player {example['player_id']}: Frame {example['from_frame']} -> {example['to_frame']}")
            print(f"      Distance: {example['distance']:.2f}m")
            print(f"      Position: ({example['from_pos'][0]:.2f}, {example['from_pos'][1]:.2f}) -> "
                  f"({example['to_pos'][0]:.2f}, {example['to_pos'][1]:.2f})")


def analyze_top_players(player_speeds: Dict, top_n: int = 10):
    """Analyze top N fastest players in detail."""

    print("\n" + "="*70)
    print(f"TOP {top_n} FASTEST PLAYERS")
    print("="*70)

    # Sort by max speed
    sorted_players = sorted(player_speeds.items(),
                           key=lambda x: x[1].get('max_speed_kmh', 0),
                           reverse=True)[:top_n]

    for rank, (player_id, data) in enumerate(sorted_players, 1):
        max_speed = data.get('max_speed_kmh', 0)
        avg_speed = data.get('average_speed_kmh', 0)
        total_distance = data.get('total_distance_m', 0)

        print(f"\n{rank}. Player {player_id}:")
        print(f"   Max Speed: {max_speed:.2f} km/h")
        print(f"   Avg Speed: {avg_speed:.2f} km/h")
        print(f"   Total Distance: {total_distance:.2f} m ({total_distance/1000:.2f} km)")

        # Speed classification
        if max_speed > 44:
            print(f"   [!!] Status: UNREALISTIC (exceeds human maximum)")
        elif max_speed > 38:
            print(f"   [!] Status: SUSPICIOUS (world-class elite level)")
        elif max_speed > 35:
            print(f"   [OK] Status: Elite sprint (rare but possible)")
        elif max_speed > 25:
            print(f"   [OK] Status: Normal sprint (expected)")
        else:
            print(f"   [OK] Status: Running/jogging (expected)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze speed calculation data from JSON output'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to analysis JSON file (e.g., sample_1_analysis_data.json)'
    )

    args = parser.parse_args()

    # Load JSON file
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        sys.exit(1)

    print(f"[*] Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract data
    player_speeds = data.get('player_speeds', {})
    metadata = data.get('metadata', {})

    if not player_speeds:
        print("[ERROR] No player_speeds data found in JSON!")
        sys.exit(1)

    # Print metadata
    print("\n" + "="*70)
    print("METADATA")
    print("="*70)
    print(f"Frames processed: {metadata.get('frames_processed', 'N/A')}")
    print(f"FPS: {metadata.get('fps', 'N/A')}")
    print(f"Processing time: {metadata.get('processing_time_seconds', 'N/A')}s")
    field_dims = metadata.get('field_dimensions', {})
    print(f"Field dimensions: {field_dims.get('width', 'N/A')}m × {field_dims.get('height', 'N/A')}m")

    # Run analyses
    analyze_field_coordinates(player_speeds)
    analyze_speed_distribution(player_speeds)
    analyze_distances(player_speeds)
    analyze_top_players(player_speeds)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\n[INFO] For more information, see: documentation/SPEED_CALCULATION_DEBUG_GUIDE.md")


if __name__ == "__main__":
    main()
