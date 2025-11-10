"""
Tactical Analysis Module

Provides homography transformations and field mapping for soccer analysis.
"""

from tactical_analysis.confidence_visualizer import ConfidenceVisualizer
from tactical_analysis.sports_compat import ViewTransformer, SoccerPitchConfiguration, draw_pitch

__all__ = [
    'ConfidenceVisualizer',
    'ViewTransformer',
    'SoccerPitchConfiguration',
    'draw_pitch'
]
